#!/usr/bin/env python3
"""
Phase 1.5: Test if Sinkhorn order affects full-model PPL.

This is the CRITICAL test for QEP-SINQ viability.

Test: Quantize entire model with different Sinkhorn orders and measure PPL.
- If PPL varies with order → QEP premise is valid
- If PPL is flat → QEP is dead

We'll monkey-patch the sinkhorn_log function to test different orders.
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import sys
sys.path.insert(0, '/workspace/SINQ')


def evaluate_ppl(model, tokenizer, max_samples=20, max_length=512):
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


def quantize_model_with_order(model_name, sinkhorn_order, bits=4, group_size=64):
    """
    Quantize a model using SINQ with specified Sinkhorn order.

    This patches the sinkhorn_log function to use the desired order.
    """
    import sinq.sinkhorn as sinkhorn_module
    import sinq.dual_shift as dual_shift_module

    # Store original function
    original_sinkhorn = sinkhorn_module.sinkhorn_log

    # Create patched version with fixed order
    def patched_sinkhorn(matrix, order=8, **kwargs):
        # Override order with our test value
        return original_sinkhorn(matrix, order=sinkhorn_order, **kwargs)

    # Patch both modules
    sinkhorn_module.sinkhorn_log = patched_sinkhorn
    dual_shift_module.sinkhorn_log = patched_sinkhorn

    try:
        # Now quantize the model
        from sinq.patch_model import AutoSINQHFModel
        from sinq.sinqlinear import sinq_base_quant_config

        config = sinq_base_quant_config(
            nbits=bits,
            group_size=group_size,
            quant_zero=False,
            quant_scale=False,
            axis=1,
            tiling_mode='1D',
            method='sinq'
        )

        print(f"    Loading and quantizing with Sinkhorn order={sinkhorn_order}...")
        quantized = AutoSINQHFModel.from_pretrained(
            model_name,
            quantization_config=config,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        return quantized

    finally:
        # Restore original function
        sinkhorn_module.sinkhorn_log = original_sinkhorn
        dual_shift_module.sinkhorn_log = original_sinkhorn


def main():
    print("=" * 70)
    print("Phase 1.5: Sinkhorn Order vs Full-Model PPL")
    print("=" * 70)
    print("\nCRITICAL TEST: Does Sinkhorn iteration count affect PPL?")
    print("If PPL is flat → QEP-SINQ is DEAD\n")

    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-1.7B"

    # First get FP16 baseline
    print("Loading FP16 baseline...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model_fp16.eval()

    print("Evaluating FP16 baseline...")
    baseline_ppl = evaluate_ppl(model_fp16, tokenizer, max_samples=15, max_length=256)
    print(f"FP16 Baseline PPL: {baseline_ppl:.2f}")

    # Free memory
    del model_fp16
    torch.cuda.empty_cache()

    # Test different Sinkhorn orders
    orders_to_test = [1, 2, 4, 8, 16, 32]
    results = {}

    for order in orders_to_test:
        print(f"\n{'=' * 70}")
        print(f"Testing Sinkhorn order = {order}")
        print("=" * 70)

        try:
            model_q = quantize_model_with_order(
                model_name,
                sinkhorn_order=order,
                bits=4,
                group_size=64
            )
            model_q.eval()

            ppl = evaluate_ppl(model_q, tokenizer, max_samples=15, max_length=256)
            results[order] = ppl
            print(f"  Order {order}: PPL = {ppl:.2f}")

            # Free memory
            del model_q
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Order {order}: FAILED - {e}")
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
        delta = ppl - baseline_ppl
        print(f"{order:<10}{ppl:<15.2f}{delta:+.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    valid_ppls = [p for p in results.values() if p < float('inf')]

    if len(valid_ppls) < 2:
        print("\nInsufficient valid results for analysis.")
        return

    ppl_range = max(valid_ppls) - min(valid_ppls)
    ppl_cv = np.std(valid_ppls) / np.mean(valid_ppls)

    # Check trend: does more iterations help?
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

    # Criteria for QEP viability:
    # 1. PPL should vary by at least 0.3 (measurable effect)
    # 2. OR there should be a clear trend

    if ppl_range > 0.3:
        print(f"\n✓ PASS: Sinkhorn order affects PPL (range = {ppl_range:.2f})")
        print("  QEP-SINQ premise is VALID.")

        if trend < -0.2:
            print("  More iterations → lower PPL (expected behavior)")
            print("  → Allocating more iterations to important layers should help.")
        elif trend > 0.2:
            print("  WARNING: More iterations → higher PPL (unexpected)")
            print("  → May need to REDUCE iterations on important layers.")
        else:
            print("  No clear trend - optimal order may be in the middle.")

        print("\n  Proceed to Phase 2: Hypothesis Formulation")

    elif ppl_cv > 0.01:
        print(f"\n✓ CONDITIONAL PASS: Sinkhorn order has small but measurable effect")
        print(f"  CV = {ppl_cv:.3f}")
        print("  Proceed with caution - effect may be too small for practical benefit.")

    else:
        print(f"\n✗ FAIL: Sinkhorn order has minimal effect on PPL")
        print(f"  PPL range: {ppl_range:.2f}")
        print(f"  PPL CV: {ppl_cv:.3f}")
        print("\n  QEP-SINQ approach is INVALID.")
        print("  Sinkhorn optimization is already converged - more iterations don't help.")
        print("\n  ABORT: Return to Phase 1.1 with new approach")

    return {
        "baseline_ppl": baseline_ppl,
        "results": results,
        "ppl_range": ppl_range,
        "ppl_cv": ppl_cv,
        "trend": trend,
        "best_order": best_order,
    }


if __name__ == "__main__":
    main()
