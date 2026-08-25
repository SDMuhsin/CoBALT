#!/usr/bin/env python3
"""
Phase 1.5: Test if Sinkhorn order affects full-model PPL.

CRITICAL TEST for QEP-SINQ viability.

Uses the proper SINQ quantization pipeline but patches the Sinkhorn order.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys
import gc
sys.path.insert(0, '/workspace/SINQ')

# Seed for reproducibility
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def evaluate_ppl(model, tokenizer, max_samples=15, max_length=256):
    """Evaluate perplexity on WikiText-2."""
    device = next(model.parameters()).device

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [s["text"] for s in dataset if len(s["text"].strip()) > 100][:max_samples]

    all_text = "\n\n".join(texts)
    encodings = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=max_length * max_samples)
    input_ids = encodings.input_ids.to(device)

    nlls = []
    stride = max_length // 2

    with torch.no_grad():
        for begin in tqdm(range(0, min(input_ids.size(1), max_length * 10), stride), desc="Evaluating", leave=False):
            end = min(begin + max_length, input_ids.size(1))
            chunk = input_ids[:, begin:end]

            outputs = model(chunk, labels=chunk)
            nll = outputs.loss.item()

            if not np.isnan(nll) and not np.isinf(nll):
                nlls.append(nll)

            if end >= input_ids.size(1):
                break

    return np.exp(np.mean(nlls)) if nlls else float('inf')


def quantize_with_order(model_name, tokenizer, sinkhorn_order, device='cuda:0'):
    """
    Quantize a model using SINQ with specified Sinkhorn order.
    """
    import sinq.dual_shift as dual_shift_module

    # Patch the sinkhorn_log call in dual_shift to use our order
    original_quantize_dual_scale_shift = dual_shift_module.quantize_dual_scale_shift

    def patched_quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
        """Patched version that uses our Sinkhorn order."""
        from sinq.sinkhorn import sinkhorn_log

        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        # Use our specified order instead of hardcoded 16
        matrix_normalized, mu1, mu2 = sinkhorn_log(matrix, sinkhorn_order)

        if not ('sinq' in method):
            matrix_normalized = matrix_normalized * mu1 * mu2
            mu1 = torch.ones_like(mu1)
            mu2 = torch.ones_like(mu2)

        if 'awq' in method:
            matrix_normalized = matrix_normalized * awq_scale
            mu1 = mu1 / awq_scale.float()

        # Standard quantization (simplified - just use uniform)
        max_val = matrix_normalized.amax(dim=1, keepdim=True)
        min_val = matrix_normalized.amin(dim=1, keepdim=True)
        max_int = min_max[1]
        min_int = min_max[0]
        scales = (max_val - min_val).clamp(min=1e-4) / max_int
        zeros = -torch.round(min_val / scales)
        q = torch.clamp(torch.round(matrix_normalized / scales + zeros), min_int, max_int).to(torch.int8)

        scales2 = torch.ones(1, matrix.shape[1]).to(dev).to(mu1.dtype) * mu1
        scales = scales * mu2

        q = q.to(dtype).to(dev)
        s1 = scales.to(dtype)
        s2 = scales2.to(dtype)
        z = zeros.to(dtype).to(dev)

        return q, s1.to(dev), s2.to(dev), z

    # Apply patch
    dual_shift_module.quantize_dual_scale_shift = patched_quantize_dual_scale_shift

    try:
        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        from sinq.patch_model import AutoSINQHFModel
        from sinq.sinqlinear import BaseQuantizeConfig

        quant_config = BaseQuantizeConfig(
            nbits=4,
            group_size=64,
            axis=1,
            tiling_mode='1D',
            method='sinq_nogemlite'  # Use PyTorch backend for consistent comparison
        )

        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device=device
        )

        return model

    finally:
        # Restore original function
        dual_shift_module.quantize_dual_scale_shift = original_quantize_dual_scale_shift


def main():
    print("=" * 70)
    print("Phase 1.5: Sinkhorn Order vs Full-Model PPL (v2)")
    print("=" * 70)
    print("\nCRITICAL TEST: Does Sinkhorn iteration count affect PPL?")
    print("If PPL is flat → QEP-SINQ is DEAD\n")

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0'

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # First get FP16 baseline
    print("Loading FP16 baseline...")
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_fp16.eval()

    print("Evaluating FP16 baseline...")
    baseline_ppl = evaluate_ppl(model_fp16, tokenizer)
    print(f"FP16 Baseline PPL: {baseline_ppl:.2f}")

    # Free memory
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # Test different Sinkhorn orders
    orders_to_test = [1, 2, 4, 8, 16, 32]
    results = {}

    for order in orders_to_test:
        print(f"\n{'=' * 70}")
        print(f"Testing Sinkhorn order = {order}")
        print("=" * 70)

        try:
            model_q = quantize_with_order(model_name, tokenizer, order, device)
            model_q.eval()

            # Move to appropriate device if needed
            if not hasattr(model_q, 'hf_device_map'):
                model_q = model_q.to(device)

            ppl = evaluate_ppl(model_q, tokenizer)
            results[order] = ppl
            print(f"  Order {order}: PPL = {ppl:.2f}")

            # Free memory
            del model_q
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"  Order {order}: FAILED")
            print(f"    Error: {e}")
            traceback.print_exc()
            results[order] = float('inf')

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nFP16 Baseline PPL: {baseline_ppl:.2f}")
    print(f"\n4-bit SINQ PPL by Sinkhorn order:")
    print(f"{'Order':<10}{'PPL':<15}{'Δ from FP16':<15}")
    print("-" * 40)

    for order in orders_to_test:
        ppl = results[order]
        if ppl < float('inf'):
            delta = ppl - baseline_ppl
            print(f"{order:<10}{ppl:<15.2f}{delta:+.2f}")
        else:
            print(f"{order:<10}{'inf':<15}{'N/A':<15}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    valid_ppls = [p for p in results.values() if p < float('inf')]

    if len(valid_ppls) < 2:
        print("\nInsufficient valid results for analysis.")
        print("Check for errors in quantization.")
        return

    ppl_range = max(valid_ppls) - min(valid_ppls)
    ppl_cv = np.std(valid_ppls) / np.mean(valid_ppls)

    # Check trend
    sorted_results = [(o, results[o]) for o in sorted(results.keys()) if results[o] < float('inf')]
    if len(sorted_results) >= 2:
        first_ppl = sorted_results[0][1]
        last_ppl = sorted_results[-1][1]
        trend = last_ppl - first_ppl
    else:
        trend = 0

    print(f"\nPPL range: {ppl_range:.2f}")
    print(f"PPL CV: {ppl_cv:.3f}")
    print(f"Trend (1→{orders_to_test[-1]}): {'+' if trend > 0 else ''}{trend:.2f}")

    # Find best order
    best_order = min(results, key=results.get)
    best_ppl = results[best_order]
    print(f"\nBest order: {best_order} (PPL = {best_ppl:.2f})")

    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    if ppl_range > 0.3:
        print(f"\n✓ PASS: Sinkhorn order affects PPL (range = {ppl_range:.2f})")
        print("  QEP-SINQ premise is VALID.")
        print("\n  Proceed to Phase 2: Hypothesis Formulation")
    elif ppl_cv > 0.01:
        print(f"\n✓ CONDITIONAL PASS: Small but measurable effect (CV = {ppl_cv:.3f})")
    else:
        print(f"\n✗ FAIL: Sinkhorn order has minimal effect on PPL")
        print(f"  PPL range: {ppl_range:.2f}, CV: {ppl_cv:.3f}")
        print("\n  QEP-SINQ approach is INVALID.")
        print("  ABORT: Return to Phase 1.1")


if __name__ == "__main__":
    main()
