#!/usr/bin/env python3
"""
Test EAS-SINQ: Error-Adaptive Sinkhorn for SINQ.

Hypothesis: Instead of pre-computing importance, adaptively increase
Sinkhorn iterations when reconstruction error is high.

Algorithm:
1. Run initial Sinkhorn iterations (e.g., 4)
2. Quantize and measure reconstruction error
3. If error > threshold, run more iterations (up to max)
4. Repeat until error is acceptable or max iterations reached

This directly responds to actual quantization difficulty.
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

# Seed
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


def eas_sinkhorn_log(matrix, min_order=4, max_order=32, error_threshold=0.1,
                     clip_min=1e-3, clip_max=1e3, eps=1e-6):
    """
    Error-Adaptive Sinkhorn: increase iterations until error is acceptable.

    Args:
        matrix: Input weight matrix
        min_order: Starting iterations
        max_order: Maximum iterations
        error_threshold: Target relative reconstruction error
    """
    from sinq.sinkhorn import sinkhorn_log

    # Track statistics
    order_used = min_order
    best_error = float('inf')
    best_result = None

    for order in range(min_order, max_order + 1, 4):  # Step by 4
        # Run Sinkhorn
        matrix_norm, mu1, mu2 = sinkhorn_log(matrix, order)

        # Quantize (simplified) to measure error
        max_val = matrix_norm.amax(dim=1, keepdim=True)
        min_val = matrix_norm.amin(dim=1, keepdim=True)
        scale = (max_val - min_val).clamp(min=1e-4) / 15  # 4-bit

        # Quantize and dequantize
        q = torch.clamp(torch.round((matrix_norm - min_val) / scale), 0, 15)
        recon = q * scale + min_val

        # Scale back to original space
        recon_orig = recon * mu1 * mu2

        # Compute relative error
        error = (matrix - recon_orig).abs().mean() / (matrix.abs().mean() + 1e-8)

        if error < best_error:
            best_error = error
            best_result = (matrix_norm, mu1, mu2)
            order_used = order

        # Early stop if error is acceptable
        if error < error_threshold:
            break

    return best_result[0], best_result[1], best_result[2], order_used, best_error.item()


def quantize_with_eas(model_name, tokenizer, error_threshold=0.1, device='cuda:0'):
    """
    Quantize using EAS-SINQ with adaptive Sinkhorn iterations.
    """
    import sinq.dual_shift as dual_shift_module
    from sinq.sinkhorn import sinkhorn_log

    original_quantize = dual_shift_module.quantize_dual_scale_shift

    # Track iteration statistics
    order_stats = []

    def patched_quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
        """Patched version with EAS."""
        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        # Use EAS Sinkhorn
        matrix_normalized, mu1, mu2, order_used, error = eas_sinkhorn_log(
            matrix,
            min_order=4,
            max_order=32,
            error_threshold=error_threshold
        )

        order_stats.append((order_used, error))

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

        # Print iteration statistics
        orders = [o for o, _ in order_stats]
        errors = [e for _, e in order_stats]
        print(f"  Tiles quantized: {len(order_stats)}")
        print(f"  Order distribution: min={min(orders)}, max={max(orders)}, mean={np.mean(orders):.1f}")
        print(f"  Error distribution: min={min(errors):.4f}, max={max(errors):.4f}, mean={np.mean(errors):.4f}")

        return model

    finally:
        dual_shift_module.quantize_dual_scale_shift = original_quantize


def quantize_standard(model_name, tokenizer, order=16, device='cuda:0'):
    """Quantize using standard SINQ."""
    import sinq.dual_shift as dual_shift_module
    from sinq.sinkhorn import sinkhorn_log

    original_quantize = dual_shift_module.quantize_dual_scale_shift

    def patched_quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        matrix_normalized, mu1, mu2 = sinkhorn_log(matrix, order)

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

        return model

    finally:
        dual_shift_module.quantize_dual_scale_shift = original_quantize


def main():
    print("=" * 70)
    print("EAS-SINQ: Error-Adaptive Sinkhorn Test")
    print("=" * 70)
    print("\nHypothesis: Adaptively increase iterations based on actual error,")
    print("not pre-computed importance.\n")

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # FP16 baseline
    print("=" * 70)
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

    # Standard SINQ (order=16)
    print("\n" + "=" * 70)
    print("Standard SINQ (order=16)")
    print("=" * 70)

    model_standard = quantize_standard(model_name, tokenizer, order=16, device=device)
    model_standard.eval()
    if not hasattr(model_standard, 'hf_device_map'):
        model_standard = model_standard.to(device)
    standard_ppl = evaluate_ppl(model_standard, tokenizer)
    print(f"Standard SINQ PPL: {standard_ppl:.2f}")
    del model_standard
    gc.collect()
    torch.cuda.empty_cache()

    # EAS-SINQ with different thresholds
    print("\n" + "=" * 70)
    print("EAS-SINQ (Error-Adaptive)")
    print("=" * 70)

    thresholds = [0.05, 0.10, 0.15]
    eas_results = {}

    for threshold in thresholds:
        print(f"\nTesting threshold = {threshold}")

        model_eas = quantize_with_eas(model_name, tokenizer, error_threshold=threshold, device=device)
        model_eas.eval()
        if not hasattr(model_eas, 'hf_device_map'):
            model_eas = model_eas.to(device)

        eas_ppl = evaluate_ppl(model_eas, tokenizer)
        eas_results[threshold] = eas_ppl
        print(f"  EAS-SINQ (threshold={threshold}): PPL = {eas_ppl:.2f}")

        del model_eas
        gc.collect()
        torch.cuda.empty_cache()

    # Results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Configuration':<35}{'PPL':<12}{'Δ from Standard':<15}")
    print("-" * 65)
    print(f"{'FP16 Baseline':<35}{baseline_ppl:<12.2f}{baseline_ppl - standard_ppl:+.2f}")
    print(f"{'Standard SINQ (order=16)':<35}{standard_ppl:<12.2f}{0:+.2f}")

    for threshold, ppl in eas_results.items():
        print(f"{'EAS-SINQ (threshold=' + str(threshold) + ')':<35}{ppl:<12.2f}{ppl - standard_ppl:+.2f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best_threshold = min(eas_results, key=eas_results.get)
    best_ppl = eas_results[best_threshold]

    if best_ppl < standard_ppl - 0.1:
        print(f"\n✓ SUCCESS: EAS-SINQ (threshold={best_threshold}) improves PPL by {standard_ppl - best_ppl:.2f}")
        print("  Error-adaptive iterations help!")
    elif abs(best_ppl - standard_ppl) <= 0.1:
        print(f"\n≈ EQUIVALENT: EAS-SINQ ≈ Standard SINQ (best Δ = {best_ppl - standard_ppl:.2f})")
    else:
        print(f"\n✗ FAILURE: EAS-SINQ is worse by {best_ppl - standard_ppl:.2f} PPL")


if __name__ == "__main__":
    main()
