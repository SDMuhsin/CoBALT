"""
Phase 1.2: Analyze error propagation across autoregressive generation steps.

Key Questions:
1. How does quantization error grow across T generation steps?
2. Is growth linear, exponential, or does it saturate?
3. Which layers contribute most to error accumulation?
4. Does the error pattern differ between quantized and FP models?

Methodology:
- Use teacher forcing with FP model's tokens to isolate quantization error from token divergence
- Measure KL divergence at each step between FP and quantized logits
- Track how error evolves: step 1, 10, 20, 50, 100
"""

import os
import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm

from sinq.patch_model import AutoSINQHFModel
from sinq.sinqlinear import BaseQuantizeConfig


def kl_divergence(p_logits, q_logits, dim=-1):
    """Compute KL(P||Q) from logits."""
    p = F.softmax(p_logits, dim=dim)
    log_p = F.log_softmax(p_logits, dim=dim)
    log_q = F.log_softmax(q_logits, dim=dim)
    kl = (p * (log_p - log_q)).sum(dim=dim)
    return kl


def js_divergence(p_logits, q_logits, dim=-1):
    """Compute JS divergence (symmetric) from logits."""
    p = F.softmax(p_logits, dim=dim)
    q = F.softmax(q_logits, dim=dim)
    m = 0.5 * (p + q)
    log_p = F.log_softmax(p_logits, dim=dim)
    log_q = F.log_softmax(q_logits, dim=dim)
    log_m = torch.log(m + 1e-10)
    kl_pm = (p * (log_p - log_m)).sum(dim=dim)
    kl_qm = (q * (log_q - log_m)).sum(dim=dim)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def analyze_error_propagation_teacher_forcing(
    model_fp, model_q, tokenizer, prompt, max_steps=100, device='cuda'
):
    """
    Analyze error propagation using teacher forcing.

    Both models receive the SAME input sequence (from FP model's greedy generation).
    This isolates quantization error from token divergence effects.
    """
    # First, generate with FP model
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']

    # Generate reference sequence with FP model
    generated_ids = model_fp.generate(
        input_ids,
        max_new_tokens=max_steps,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    # Now measure error at each step using teacher forcing
    kl_per_step = []
    js_per_step = []

    prompt_len = input_ids.shape[1]

    for step in range(max_steps):
        # Get the sequence up to this step
        seq_len = prompt_len + step
        current_ids = generated_ids[:, :seq_len]

        # Get logits from both models for the next token
        outputs_fp = model_fp(current_ids)
        outputs_q = model_q(current_ids)

        # Only look at the last position's logits
        logits_fp = outputs_fp.logits[:, -1, :]
        logits_q = outputs_q.logits[:, -1, :]

        # Compute divergence
        kl = kl_divergence(logits_fp, logits_q).mean().item()
        js = js_divergence(logits_fp, logits_q).mean().item()

        kl_per_step.append(kl)
        js_per_step.append(js)

    return kl_per_step, js_per_step


@torch.no_grad()
def analyze_error_propagation_free_generation(
    model_fp, model_q, tokenizer, prompt, max_steps=100, device='cuda'
):
    """
    Analyze error with free generation (each model uses its own tokens).

    This captures the full effect: quantization error + token divergence.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']
    prompt_len = input_ids.shape[1]

    # Generate with both models
    gen_fp = model_fp.generate(
        input_ids.clone(),
        max_new_tokens=max_steps,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )
    gen_q = model_q.generate(
        input_ids.clone(),
        max_new_tokens=max_steps,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    # Count token matches at each step
    token_matches = []
    for step in range(max_steps):
        idx = prompt_len + step
        if idx < gen_fp.shape[1] and idx < gen_q.shape[1]:
            match = (gen_fp[0, idx] == gen_q[0, idx]).item()
            token_matches.append(match)
        else:
            token_matches.append(0)

    # Calculate cumulative accuracy
    cumulative_accuracy = []
    for i in range(len(token_matches)):
        acc = sum(token_matches[:i+1]) / (i + 1)
        cumulative_accuracy.append(acc)

    return token_matches, cumulative_accuracy


@torch.no_grad()
def analyze_layer_contributions(
    model_fp, model_q, tokenizer, prompt, step=50, device='cuda'
):
    """
    Analyze which layers contribute most to error at a given generation step.

    Method: For each layer, use FP weights for that layer and quantized for others.
    Measure how much error decreases when we "fix" that layer.
    """
    # This is expensive, so just return None for now
    # Would require model surgery which is complex
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nbits", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    print("="*70)
    print("PHASE 1.2: ANALYZE ERROR PROPAGATION ACROSS GENERATION STEPS")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Bits: {args.nbits}")
    print(f"Max steps: {args.max_steps}")

    # Load FP model
    print("\nLoading FP model...")
    model_fp = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and quantize model
    print("\nLoading and quantizing model with SINQ...")
    model_q = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )

    quant_config = BaseQuantizeConfig(
        nbits=args.nbits,
        group_size=64,
        axis=1,
        tiling_mode='1D',
        method='sinq_nogemlite'
    )

    AutoSINQHFModel.quantize_model(
        model_q,
        tokenizer=tokenizer,
        quant_config=quant_config,
        compute_dtype=torch.float16,
        device=args.device
    )

    # Test prompts
    prompts = [
        "The quick brown fox",
        "In the beginning of the universe,",
        "Machine learning is a field of artificial intelligence that",
        "The capital of France is Paris. The capital of Germany is",
    ]

    print("\n" + "="*70)
    print("ANALYSIS 1: ERROR PROPAGATION WITH TEACHER FORCING")
    print("(Both models see same input - isolates quantization error)")
    print("="*70)

    all_kl = []
    all_js = []

    for prompt in prompts:
        print(f"\nPrompt: '{prompt[:50]}...'")
        kl_steps, js_steps = analyze_error_propagation_teacher_forcing(
            model_fp, model_q, tokenizer, prompt,
            max_steps=args.max_steps, device=args.device
        )
        all_kl.append(kl_steps)
        all_js.append(js_steps)

    # Average across prompts
    avg_kl = np.mean(all_kl, axis=0)
    avg_js = np.mean(all_js, axis=0)

    print("\n" + "-"*70)
    print("AVERAGE KL DIVERGENCE BY STEP (teacher forcing)")
    print("-"*70)
    steps_to_report = [1, 5, 10, 20, 50, 100]
    for s in steps_to_report:
        if s <= len(avg_kl):
            print(f"Step {s:3d}: KL={avg_kl[s-1]:.6f}, JS={avg_js[s-1]:.6f}")

    # Analyze growth pattern
    print("\n" + "-"*70)
    print("ERROR GROWTH ANALYSIS")
    print("-"*70)

    kl_1 = avg_kl[0]
    kl_10 = avg_kl[9] if len(avg_kl) >= 10 else avg_kl[-1]
    kl_50 = avg_kl[49] if len(avg_kl) >= 50 else avg_kl[-1]
    kl_100 = avg_kl[99] if len(avg_kl) >= 100 else avg_kl[-1]

    print(f"KL at step 1: {kl_1:.6f}")
    print(f"KL at step 10: {kl_10:.6f} (ratio to step 1: {kl_10/kl_1:.2f}x)")
    print(f"KL at step 50: {kl_50:.6f} (ratio to step 1: {kl_50/kl_1:.2f}x)")
    print(f"KL at step 100: {kl_100:.6f} (ratio to step 1: {kl_100/kl_1:.2f}x)")

    # Determine growth pattern
    # Linear: ratio ≈ step number
    # Exponential: ratio >> step number
    # Saturating: ratio << step number

    ratio_10 = kl_10 / kl_1
    ratio_50 = kl_50 / kl_1
    ratio_100 = kl_100 / kl_1

    if ratio_100 < 2:
        pattern = "SATURATING (error doesn't grow much)"
    elif ratio_100 > 50:
        pattern = "EXPONENTIAL (error grows very fast)"
    else:
        pattern = "APPROXIMATELY LINEAR (error grows proportionally)"

    print(f"\nGrowth pattern: {pattern}")

    print("\n" + "="*70)
    print("ANALYSIS 2: TOKEN DIVERGENCE WITH FREE GENERATION")
    print("(Each model uses its own tokens)")
    print("="*70)

    for prompt in prompts[:2]:  # Just first 2 for speed
        print(f"\nPrompt: '{prompt[:50]}...'")
        matches, cum_acc = analyze_error_propagation_free_generation(
            model_fp, model_q, tokenizer, prompt,
            max_steps=args.max_steps, device=args.device
        )

        print(f"Token match rate at step 10: {cum_acc[9]:.2%}")
        print(f"Token match rate at step 50: {cum_acc[49]:.2%}")
        print(f"Token match rate at step 100: {cum_acc[99] if len(cum_acc) >= 100 else cum_acc[-1]:.2%}")

    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print(f"""
1. Error Growth Pattern: {pattern}
2. KL divergence at step 1 vs step 100: {kl_1:.6f} -> {kl_100:.6f} ({ratio_100:.2f}x)

IMPLICATIONS FOR GENERATION-AWARE OPTIMIZATION:
""")

    if ratio_100 < 2:
        print("""
- Error is relatively STABLE across generation steps
- This suggests layer-reconstruction optimization may already work well
- Generation-aware optimization may not provide significant benefit
- CAUTION: Gap 4 hypothesis may be weak
""")
    elif ratio_100 > 10:
        print("""
- Error ACCUMULATES significantly across generation steps
- This validates the core premise of Gap 4
- Generation-aware optimization could help by identifying and fixing
  layers that contribute most to error propagation
- OPPORTUNITY: Gap 4 hypothesis is promising
""")
    else:
        print("""
- Error growth is MODERATE
- There may be opportunity for generation-aware optimization
- Need to investigate which layers contribute most to accumulation
""")


if __name__ == "__main__":
    main()
