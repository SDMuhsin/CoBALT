#!/usr/bin/env python3
"""
Phase 6: Evaluate PAKV on WikiText-2 Perplexity

This is the critical evaluation: Does the MSE improvement from
PAKV (sink token strategy) translate to actual perplexity improvement?

Key comparison:
- Uniform 4-bit KV cache
- PAKV (6-4 sink strategy, 4.2 avg bits)
- Uniform 5-bit KV cache

Success criterion: PAKV at 4.2 bits should have lower PPL than
Uniform 4-bit and ideally match or beat Uniform 5-bit.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, '/workspace/SINQ')
from sinq.pakv import PAKVQuantizer, PAKVConfig


def quantize_tensor(x, num_bits, per_channel=True):
    """Simple min-max quantization."""
    if per_channel:
        x_min = x.min(dim=-1, keepdim=True).values
        x_max = x.max(dim=-1, keepdim=True).values
    else:
        x_min = x.min()
        x_max = x.max()

    range_val = x_max - x_min
    range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)

    qmax = 2**num_bits - 1
    scale = range_val / qmax
    x_q = torch.round((x - x_min) / scale) * scale + x_min

    return x_q


def quantize_kv_uniform(kv_cache, num_bits):
    """Uniform quantization of entire KV cache."""
    quantized = DynamicCache()

    for layer_idx in range(len(kv_cache)):
        k = kv_cache.key_cache[layer_idx]
        v = kv_cache.value_cache[layer_idx]

        k_q = quantize_tensor(k, num_bits)
        v_q = quantize_tensor(v, num_bits)

        quantized.update(k_q, v_q, layer_idx)

    return quantized


def evaluate_ppl_with_kv_quantization(
    model,
    tokenizer,
    dataset,
    quant_fn,
    max_samples=100,
    max_seq_len=1024,
    stride=512
):
    """
    Evaluate perplexity with KV cache quantization.

    Uses a sliding window approach to handle long sequences.
    At each step, the KV cache from the context is quantized
    before computing the next-token probabilities.
    """
    device = next(model.parameters()).device

    nlls = []
    total_tokens = 0

    for i, sample in enumerate(tqdm(dataset, total=max_samples, desc="Evaluating")):
        if i >= max_samples:
            break

        text = sample["text"]
        if len(text.strip()) == 0:
            continue

        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.shape[1]

        if seq_len < 2:
            continue

        # Compute NLL for this sequence
        with torch.no_grad():
            # Full forward pass to get KV cache
            outputs = model(input_ids, use_cache=True)
            fp_kv = outputs.past_key_values

            # Quantize KV cache
            q_kv = quant_fn(fp_kv)

            # Compute per-token NLL using quantized KV cache
            # We need to recompute logits with quantized KV

            # For fair evaluation, we measure how well the quantized KV
            # predicts the next token at each position
            # This requires a careful per-token evaluation

            # Simple approach: Get logits from the quantized model
            # by doing inference with the quantized KV cache

            # We'll evaluate on the last N tokens where we have enough context
            eval_start = min(64, seq_len // 2)  # Start evaluating after some context

            sample_nlls = []
            for pos in range(eval_start, seq_len - 1):
                # Get context up to pos
                context_ids = input_ids[:, :pos+1]

                # Forward to get KV
                context_outputs = model(context_ids, use_cache=True)
                context_kv = context_outputs.past_key_values

                # Quantize context KV
                q_context_kv = quant_fn(context_kv)

                # Get prediction for next token using quantized KV
                dummy_input = input_ids[:, pos:pos+1]  # Current token
                pred_outputs = model(
                    input_ids=dummy_input,
                    past_key_values=q_context_kv,
                    use_cache=False
                )
                logits = pred_outputs.logits[:, -1, :]

                # Get target token
                target = input_ids[:, pos+1]

                # Compute NLL
                log_probs = F.log_softmax(logits, dim=-1)
                nll = -log_probs[0, target].item()
                sample_nlls.append(nll)

            if sample_nlls:
                nlls.extend(sample_nlls)
                total_tokens += len(sample_nlls)

    if not nlls:
        return float('inf')

    avg_nll = np.mean(nlls)
    ppl = np.exp(avg_nll)

    return ppl


def evaluate_ppl_simple(
    model,
    tokenizer,
    dataset,
    quant_fn,
    max_samples=50,
    context_len=512,
):
    """
    Simplified PPL evaluation.

    For each sample:
    1. Encode the full sequence
    2. Use first context_len tokens as context
    3. Quantize KV cache for context
    4. Evaluate next-token prediction for remaining tokens
    """
    device = next(model.parameters()).device

    all_nlls = []

    for i, sample in enumerate(tqdm(dataset, total=max_samples, desc="Evaluating")):
        if i >= max_samples:
            break

        text = sample["text"]
        if len(text.strip()) == 0:
            continue

        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=context_len + 64)
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.shape[1]

        if seq_len < context_len + 10:
            continue

        with torch.no_grad():
            # Get KV cache for context
            context_ids = input_ids[:, :context_len]
            context_outputs = model(context_ids, use_cache=True)
            fp_kv = context_outputs.past_key_values

            # Quantize KV cache
            q_kv = quant_fn(fp_kv)

            # Evaluate on remaining tokens
            for pos in range(context_len, seq_len - 1):
                # Get the token at this position
                curr_token = input_ids[:, pos:pos+1]
                target_token = input_ids[0, pos + 1].item()

                # Predict next token using quantized KV
                pred_outputs = model(
                    input_ids=curr_token,
                    past_key_values=q_kv,
                    use_cache=True  # Update KV cache
                )
                logits = pred_outputs.logits[:, -1, :]
                q_kv = pred_outputs.past_key_values  # Get updated KV

                # Re-quantize the updated KV cache
                q_kv = quant_fn(q_kv)

                # Compute NLL
                log_probs = F.log_softmax(logits, dim=-1)
                nll = -log_probs[0, target_token].item()
                all_nlls.append(nll)

    if not all_nlls:
        return float('inf')

    avg_nll = np.mean(all_nlls)
    ppl = np.exp(avg_nll)

    return ppl


def main():
    print("=" * 70)
    print("Phase 6: PAKV WikiText-2 Perplexity Evaluation")
    print("=" * 70)

    model_name = "Qwen/Qwen3-1.7B"
    print(f"\nLoading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Load WikiText-2
    print("\nLoading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Filter out empty samples
    dataset = [s for s in dataset if len(s["text"].strip()) > 100]
    print(f"Loaded {len(dataset)} non-empty samples")

    # Define quantization configurations
    pakv_config = PAKVConfig(sink_fraction=0.1, sink_bits=6, rest_bits=4)
    pakv_quantizer = PAKVQuantizer(pakv_config)

    configs = [
        ("FP (no quant)", lambda kv: kv, 16.0),
        ("Uniform 3-bit", lambda kv: quantize_kv_uniform(kv, 3), 3.0),
        ("Uniform 4-bit", lambda kv: quantize_kv_uniform(kv, 4), 4.0),
        ("PAKV 6-4", pakv_quantizer.quantize_kv_cache, 4.2),
        ("Uniform 5-bit", lambda kv: quantize_kv_uniform(kv, 5), 5.0),
        ("Uniform 8-bit", lambda kv: quantize_kv_uniform(kv, 8), 8.0),
    ]

    print("\n" + "=" * 70)
    print("Evaluating Perplexity")
    print("=" * 70)

    results = []
    max_samples = 30  # Use fewer samples for faster evaluation

    for name, quant_fn, avg_bits in configs:
        print(f"\nEvaluating: {name} ({avg_bits} bits)...")
        ppl = evaluate_ppl_simple(model, tokenizer, dataset, quant_fn, max_samples=max_samples)
        results.append({
            "name": name,
            "avg_bits": avg_bits,
            "ppl": ppl,
        })
        print(f"  PPL = {ppl:.2f}")

    # Print comparison table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Config':<20} | {'Avg Bits':>10} | {'Perplexity':>12}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<20} | {r['avg_bits']:>10.1f} | {r['ppl']:>12.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    fp_ppl = next(r for r in results if "FP" in r["name"])["ppl"]
    uniform_4bit = next(r for r in results if r["name"] == "Uniform 4-bit")
    pakv = next(r for r in results if r["name"] == "PAKV 6-4")
    uniform_5bit = next(r for r in results if r["name"] == "Uniform 5-bit")

    print(f"\n1. PAKV vs Uniform 4-bit:")
    print(f"   PAKV PPL: {pakv['ppl']:.2f}")
    print(f"   Uniform 4-bit PPL: {uniform_4bit['ppl']:.2f}")
    if pakv['ppl'] < uniform_4bit['ppl']:
        improvement = (uniform_4bit['ppl'] - pakv['ppl']) / uniform_4bit['ppl'] * 100
        print(f"   -> PAKV is {improvement:.1f}% BETTER (lower PPL)")
    else:
        degradation = (pakv['ppl'] - uniform_4bit['ppl']) / uniform_4bit['ppl'] * 100
        print(f"   -> PAKV is {degradation:.1f}% WORSE")

    print(f"\n2. PAKV vs Uniform 5-bit:")
    print(f"   PAKV PPL: {pakv['ppl']:.2f} (4.2 bits)")
    print(f"   Uniform 5-bit PPL: {uniform_5bit['ppl']:.2f} (5.0 bits)")
    if pakv['ppl'] <= uniform_5bit['ppl']:
        memory_savings = (5.0 - 4.2) / 5.0 * 100
        print(f"   -> PAKV MATCHES/BEATS Uniform 5-bit with {memory_savings:.0f}% memory savings!")
        print("   -> PUBLICATION-WORTHY RESULT")
    else:
        gap = (pakv['ppl'] - uniform_5bit['ppl'])
        print(f"   -> PAKV is {gap:.2f} PPL higher than Uniform 5-bit")

    print(f"\n3. Distance from FP baseline:")
    print(f"   FP PPL: {fp_ppl:.2f}")
    print(f"   PAKV gap: {pakv['ppl'] - fp_ppl:.2f}")
    print(f"   Uniform 4-bit gap: {uniform_4bit['ppl'] - fp_ppl:.2f}")

    print("\n" + "=" * 70)
    print("Evaluation Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
