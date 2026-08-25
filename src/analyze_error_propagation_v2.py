"""
Phase 1.2 (continued): Deeper analysis of error propagation.

Key observation from v1: KL divergence is SATURATING (only 1.57x growth over 100 steps)
Yet token match rates are very low.

This suggests:
1. Per-step KL is noisy/variable (not consistently growing)
2. The important metric may be cumulative error, not per-step error
3. Token divergence may be the real problem, not logit divergence

New analysis:
1. Track cumulative cross-entropy loss (total bits of information lost)
2. Track top-1 token accuracy at each step
3. Analyze variance of error across steps
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


@torch.no_grad()
def analyze_detailed_error(
    model_fp, model_q, tokenizer, prompt, max_steps=100, device='cuda'
):
    """
    Detailed error analysis with multiple metrics.
    """
    # Generate reference sequence with FP model
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']

    generated_ids = model_fp.generate(
        input_ids,
        max_new_tokens=max_steps,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

    prompt_len = input_ids.shape[1]

    metrics = {
        'kl_per_step': [],
        'top1_match': [],
        'top5_match': [],
        'prob_ratio': [],  # P_q(correct_token) / P_fp(correct_token)
        'ce_loss_fp': [],
        'ce_loss_q': [],
    }

    for step in range(min(max_steps, generated_ids.shape[1] - prompt_len)):
        seq_len = prompt_len + step
        current_ids = generated_ids[:, :seq_len]

        if seq_len >= generated_ids.shape[1]:
            break

        next_token = generated_ids[0, seq_len].item()

        outputs_fp = model_fp(current_ids)
        outputs_q = model_q(current_ids)

        logits_fp = outputs_fp.logits[:, -1, :].float()
        logits_q = outputs_q.logits[:, -1, :].float()

        # KL divergence
        p_fp = F.softmax(logits_fp, dim=-1)
        p_q = F.softmax(logits_q, dim=-1)
        log_p_fp = F.log_softmax(logits_fp, dim=-1)
        log_p_q = F.log_softmax(logits_q, dim=-1)
        kl = (p_fp * (log_p_fp - log_p_q)).sum(dim=-1).item()
        metrics['kl_per_step'].append(kl)

        # Top-1 accuracy
        top1_fp = logits_fp.argmax(dim=-1).item()
        top1_q = logits_q.argmax(dim=-1).item()
        metrics['top1_match'].append(1 if top1_fp == top1_q else 0)

        # Top-5 accuracy
        top5_fp = logits_fp.topk(5, dim=-1).indices[0].tolist()
        top5_q = logits_q.topk(5, dim=-1).indices[0].tolist()
        metrics['top5_match'].append(1 if top1_fp in top5_q else 0)

        # Probability ratio for correct token
        prob_fp = p_fp[0, next_token].item()
        prob_q = p_q[0, next_token].item()
        metrics['prob_ratio'].append(prob_q / (prob_fp + 1e-10))

        # Cross-entropy loss
        ce_fp = F.cross_entropy(logits_fp, torch.tensor([next_token]).to(device)).item()
        ce_q = F.cross_entropy(logits_q, torch.tensor([next_token]).to(device)).item()
        metrics['ce_loss_fp'].append(ce_fp)
        metrics['ce_loss_q'].append(ce_q)

    return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--nbits", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=100)
    args = parser.parse_args()

    print("="*70)
    print("PHASE 1.2 (v2): DETAILED ERROR PROPAGATION ANALYSIS")
    print("="*70)

    # Load models
    print("\nLoading FP model...")
    model_fp = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog. Then",
        "In the field of machine learning, neural networks",
        "Once upon a time in a kingdom far away, there lived",
        "The fundamental theorem of calculus states that",
    ]

    print("\n" + "="*70)
    print("DETAILED ERROR ANALYSIS")
    print("="*70)

    all_metrics = []
    for prompt in prompts:
        print(f"\nAnalyzing: '{prompt[:40]}...'")
        metrics = analyze_detailed_error(
            model_fp, model_q, tokenizer, prompt,
            max_steps=args.max_steps, device=args.device
        )
        all_metrics.append(metrics)

    # Aggregate across prompts
    def avg_at_step(metric_name, step):
        values = [m[metric_name][step] for m in all_metrics if len(m[metric_name]) > step]
        return np.mean(values) if values else 0

    print("\n" + "-"*70)
    print("METRICS BY GENERATION STEP")
    print("-"*70)
    print(f"{'Step':<6} {'KL Div':>10} {'Top1 Match':>12} {'Top5 Match':>12} {'CE Loss Q':>12} {'CE Loss FP':>12}")
    print("-"*70)

    for s in [1, 5, 10, 20, 50, 100]:
        s_idx = s - 1
        if s_idx < min(len(m['kl_per_step']) for m in all_metrics):
            kl = avg_at_step('kl_per_step', s_idx)
            top1 = avg_at_step('top1_match', s_idx)
            top5 = avg_at_step('top5_match', s_idx)
            ce_q = avg_at_step('ce_loss_q', s_idx)
            ce_fp = avg_at_step('ce_loss_fp', s_idx)
            print(f"{s:<6} {kl:>10.4f} {top1:>12.2%} {top5:>12.2%} {ce_q:>12.4f} {ce_fp:>12.4f}")

    # Cumulative metrics
    print("\n" + "-"*70)
    print("CUMULATIVE METRICS")
    print("-"*70)

    for metric_name in ['top1_match', 'top5_match']:
        cum_10 = np.mean([np.mean(m[metric_name][:10]) for m in all_metrics])
        cum_50 = np.mean([np.mean(m[metric_name][:50]) for m in all_metrics])
        cum_100 = np.mean([np.mean(m[metric_name][:100]) for m in all_metrics if len(m[metric_name]) >= 100])
        print(f"{metric_name}: steps 1-10: {cum_10:.2%}, steps 1-50: {cum_50:.2%}, steps 1-100: {cum_100:.2%}")

    # Cross-entropy difference
    ce_diffs = []
    for m in all_metrics:
        for i in range(len(m['ce_loss_q'])):
            ce_diffs.append(m['ce_loss_q'][i] - m['ce_loss_fp'][i])

    print(f"\nAverage CE difference (Q - FP): {np.mean(ce_diffs):.4f} (std: {np.std(ce_diffs):.4f})")
    print(f"This corresponds to ~{np.exp(np.mean(ce_diffs)):.2f}x perplexity multiplier")

    # Growth analysis
    print("\n" + "-"*70)
    print("ERROR GROWTH TREND ANALYSIS")
    print("-"*70)

    kl_early = np.mean([np.mean(m['kl_per_step'][:10]) for m in all_metrics])
    kl_mid = np.mean([np.mean(m['kl_per_step'][40:60]) for m in all_metrics if len(m['kl_per_step']) >= 60])
    kl_late = np.mean([np.mean(m['kl_per_step'][-20:]) for m in all_metrics if len(m['kl_per_step']) >= 20])

    print(f"Average KL (steps 1-10):  {kl_early:.4f}")
    print(f"Average KL (steps 40-60): {kl_mid:.4f}")
    print(f"Average KL (last 20):     {kl_late:.4f}")

    if kl_late > kl_early * 1.5:
        trend = "INCREASING - error accumulates over time"
    elif kl_late < kl_early * 0.7:
        trend = "DECREASING - error diminishes over time"
    else:
        trend = "STABLE - error is roughly constant"

    print(f"\nTrend: {trend}")

    # Key insight
    print("\n" + "="*70)
    print("KEY INSIGHT FOR GAP 4")
    print("="*70)

    avg_top1 = np.mean([np.mean(m['top1_match']) for m in all_metrics])

    if avg_top1 > 0.9:
        print("""
Finding: Token predictions are highly consistent between FP and quantized models.
Implication: Standard layer-reconstruction optimization works well.
Gap 4 opportunity: LOW - generation quality is already good.
""")
    elif avg_top1 > 0.7:
        print("""
Finding: Token predictions differ moderately between FP and quantized models.
Implication: There is room for improvement.
Gap 4 opportunity: MEDIUM - need to identify if generation-aware optimization helps.
""")
    else:
        print("""
Finding: Token predictions differ significantly between FP and quantized models.
Implication: Quantization causes meaningful generation errors.
Gap 4 opportunity: HIGH - generation-aware optimization could help.
Key question: Is the divergence due to WHICH layers, or is it uniform?
""")


if __name__ == "__main__":
    main()
