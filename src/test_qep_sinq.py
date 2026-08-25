#!/usr/bin/env python3
"""
Phase 4-6: Test QEP-SINQ against standard SINQ.

This test:
1. Quantizes Qwen 1.7B with standard SINQ (uniform 16 iterations)
2. Quantizes Qwen 1.7B with QEP-SINQ (importance-weighted iterations)
3. Compares PPL on WikiText-2

Success criteria: QEP-SINQ PPL <= Standard SINQ PPL
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


def quantize_with_qep(model_name, tokenizer, importance, base_order=8, alpha=0.5, device='cuda:0'):
    """
    Quantize a model using QEP-SINQ with importance-weighted iterations.
    """
    import sinq.dual_shift as dual_shift_module
    from sinq.sinkhorn import sinkhorn_log
    from sinq.qep_sinq import get_qep_sinkhorn_order

    # Store original function
    original_quantize = dual_shift_module.quantize_dual_scale_shift

    # Track which layer we're quantizing
    current_layer = [0]
    layer_orders = []

    def patched_quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
        """Patched version with QEP iteration allocation."""
        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        # Get QEP order for current layer
        # Note: We increment layer counter based on number of weight matrices
        # There are 7 linear layers per transformer block
        layer_idx = current_layer[0] // 7
        order = get_qep_sinkhorn_order(layer_idx, importance, base_order, alpha)
        layer_orders.append((layer_idx, order))

        # Increment counter
        current_layer[0] += 1

        # Use QEP order
        matrix_normalized, mu1, mu2 = sinkhorn_log(matrix, order)

        if not ('sinq' in method):
            matrix_normalized = matrix_normalized * mu1 * mu2
            mu1 = torch.ones_like(mu1)
            mu2 = torch.ones_like(mu2)

        if 'awq' in method:
            matrix_normalized = matrix_normalized * awq_scale
            mu1 = mu1 / awq_scale.float()

        # Standard quantization
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
            method='sinq_nogemlite'
        )

        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device=device
        )

        # Print layer order statistics
        if layer_orders:
            order_by_layer = {}
            for layer_idx, order in layer_orders:
                if layer_idx not in order_by_layer:
                    order_by_layer[layer_idx] = []
                order_by_layer[layer_idx].append(order)

            total_iters = sum(o for _, o in layer_orders)
            print(f"  Total iterations: {total_iters}")

        return model

    finally:
        # Restore original function
        dual_shift_module.quantize_dual_scale_shift = original_quantize


def quantize_standard(model_name, tokenizer, sinkhorn_order=16, device='cuda:0'):
    """
    Quantize a model using standard SINQ with uniform iterations.
    """
    import sinq.dual_shift as dual_shift_module
    from sinq.sinkhorn import sinkhorn_log

    original_quantize = dual_shift_module.quantize_dual_scale_shift
    iter_count = [0]

    def patched_quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
        """Standard SINQ with configurable order."""
        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        iter_count[0] += 1

        matrix_normalized, mu1, mu2 = sinkhorn_log(matrix, sinkhorn_order)

        if not ('sinq' in method):
            matrix_normalized = matrix_normalized * mu1 * mu2
            mu1 = torch.ones_like(mu1)
            mu2 = torch.ones_like(mu2)

        if 'awq' in method:
            matrix_normalized = matrix_normalized * awq_scale
            mu1 = mu1 / awq_scale.float()

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

    dual_shift_module.quantize_dual_scale_shift = patched_quantize_dual_scale_shift

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        from sinq.patch_model import AutoSINQHFModel
        from sinq.sinqlinear import BaseQuantizeConfig

        quant_config = BaseQuantizeConfig(
            nbits=4,
            group_size=64,
            axis=1,
            tiling_mode='1D',
            method='sinq_nogemlite'
        )

        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device=device
        )

        print(f"  Total weight matrices quantized: {iter_count[0]}")
        print(f"  Total iterations: {iter_count[0] * sinkhorn_order}")

        return model

    finally:
        dual_shift_module.quantize_dual_scale_shift = original_quantize


def main():
    print("=" * 70)
    print("QEP-SINQ vs Standard SINQ Comparison")
    print("=" * 70)

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0'

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use pre-computed importance weights for Qwen 1.7B
    # (Normalized so mean = 1.0)
    importance = {
        0: 4.80, 1: 3.76, 2: 3.12, 3: 2.60, 4: 2.33, 5: 1.99, 6: 1.68,
        7: 1.46, 8: 1.20, 9: 0.98, 10: 0.77, 11: 0.65, 12: 0.56, 13: 0.46,
        14: 0.38, 15: 0.29, 16: 0.22, 17: 0.16, 18: 0.13, 19: 0.09,
        20: 0.07, 21: 0.06, 22: 0.05, 23: 0.05, 24: 0.04, 25: 0.04,
        26: 0.04, 27: 0.04
    }

    # Print expected allocation
    from sinq.qep_sinq import get_qep_sinkhorn_order
    print("\nQEP Iteration Allocation (base=8, alpha=0.5):")
    print("-" * 40)
    total_qep = 0
    for i in range(28):
        order = get_qep_sinkhorn_order(i, importance, base_order=8, alpha=0.5)
        total_qep += order * 7  # 7 linear layers per block
        print(f"  Layer {i:2d}: importance={importance[i]:.2f}, order={order}")
    print(f"\nTotal QEP iterations: {total_qep}")
    print(f"Uniform (order=16) would be: {28 * 7 * 16} = {28 * 7 * 16}")

    # FP16 baseline
    print("\n" + "=" * 70)
    print("FP16 Baseline")
    print("=" * 70)

    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model_fp16.eval()
    baseline_ppl = evaluate_ppl(model_fp16, tokenizer)
    print(f"FP16 PPL: {baseline_ppl:.2f}")
    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # Standard SINQ with order=16
    print("\n" + "=" * 70)
    print("Standard SINQ (uniform order=16)")
    print("=" * 70)

    model_standard = quantize_standard(model_name, tokenizer, sinkhorn_order=16, device=device)
    model_standard.eval()
    if not hasattr(model_standard, 'hf_device_map'):
        model_standard = model_standard.to(device)
    standard_ppl = evaluate_ppl(model_standard, tokenizer)
    print(f"Standard SINQ PPL: {standard_ppl:.2f}")
    del model_standard
    gc.collect()
    torch.cuda.empty_cache()

    # QEP-SINQ
    print("\n" + "=" * 70)
    print("QEP-SINQ (importance-weighted iterations)")
    print("=" * 70)

    model_qep = quantize_with_qep(model_name, tokenizer, importance, base_order=8, alpha=0.5, device=device)
    model_qep.eval()
    if not hasattr(model_qep, 'hf_device_map'):
        model_qep = model_qep.to(device)
    qep_ppl = evaluate_ppl(model_qep, tokenizer)
    print(f"QEP-SINQ PPL: {qep_ppl:.2f}")
    del model_qep
    gc.collect()
    torch.cuda.empty_cache()

    # Standard SINQ with order=8 (same base as QEP)
    print("\n" + "=" * 70)
    print("Standard SINQ (uniform order=8)")
    print("=" * 70)

    model_order8 = quantize_standard(model_name, tokenizer, sinkhorn_order=8, device=device)
    model_order8.eval()
    if not hasattr(model_order8, 'hf_device_map'):
        model_order8 = model_order8.to(device)
    order8_ppl = evaluate_ppl(model_order8, tokenizer)
    print(f"SINQ (order=8) PPL: {order8_ppl:.2f}")
    del model_order8
    gc.collect()
    torch.cuda.empty_cache()

    # Results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Configuration':<30}{'PPL':<12}{'Δ from FP16':<15}")
    print("-" * 60)
    print(f"{'FP16 Baseline':<30}{baseline_ppl:<12.2f}{0:+.2f}")
    print(f"{'SINQ (order=16)':<30}{standard_ppl:<12.2f}{standard_ppl - baseline_ppl:+.2f}")
    print(f"{'SINQ (order=8)':<30}{order8_ppl:<12.2f}{order8_ppl - baseline_ppl:+.2f}")
    print(f"{'QEP-SINQ':<30}{qep_ppl:<12.2f}{qep_ppl - baseline_ppl:+.2f}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    qep_vs_standard = standard_ppl - qep_ppl
    qep_vs_order8 = order8_ppl - qep_ppl

    print(f"\nQEP-SINQ vs Standard (order=16): {qep_vs_standard:+.2f} PPL")
    print(f"QEP-SINQ vs Uniform (order=8): {qep_vs_order8:+.2f} PPL")

    # Decision
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if qep_ppl < standard_ppl - 0.1:
        print(f"\n✓ SUCCESS: QEP-SINQ improves PPL by {standard_ppl - qep_ppl:.2f}")
        print("  Importance weighting is beneficial!")
    elif abs(qep_ppl - standard_ppl) <= 0.1:
        print(f"\n≈ EQUIVALENT: QEP-SINQ ≈ Standard SINQ (Δ = {qep_ppl - standard_ppl:.2f})")
        print("  Importance weighting preserves PPL with potentially fewer iterations.")
    else:
        print(f"\n✗ FAILURE: QEP-SINQ is worse by {qep_ppl - standard_ppl:.2f} PPL")
        print("  Importance weighting hurts performance.")


if __name__ == "__main__":
    main()
