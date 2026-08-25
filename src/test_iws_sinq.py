#!/usr/bin/env python3
"""
Test IWS-SINQ: Importance-Weighted Sinkhorn for SINQ.

This test:
1. Computes per-row/column importance weights via gradient of loss w.r.t weights
2. Modifies Sinkhorn to use importance-weighted target std
3. Compares PPL against standard SINQ

Hypothesis: Important rows/cols with lower target std get less quantization error,
improving PPL where it matters most.
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


def compute_weight_importance(model, tokenizer, texts, max_length=128, device='cuda:0'):
    """
    Compute importance of each weight element via gradient magnitude.

    Returns dict mapping layer_name -> (row_importance, col_importance)
    """
    model = model.to(device)
    model.train()  # Need gradients

    # Storage for gradient accumulation
    grad_accum = {}

    for text in tqdm(texts[:10], desc="Computing weight importance"):  # Limit samples
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(device)

        if input_ids.shape[1] < 10:
            continue

        model.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()

        # Accumulate gradients for linear layers
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) == 2:
                if param.grad is not None:
                    grad_abs = param.grad.abs().detach()
                    if name not in grad_accum:
                        grad_accum[name] = grad_abs.clone()
                    else:
                        grad_accum[name] += grad_abs

    model.eval()

    # Compute row/column importance from accumulated gradients
    importance = {}
    for name, grad_sum in grad_accum.items():
        # Row importance: mean gradient magnitude per row
        row_imp = grad_sum.mean(dim=1)  # [out_features]
        # Column importance: mean gradient magnitude per column
        col_imp = grad_sum.mean(dim=0)  # [in_features]

        # Normalize to mean 1.0
        row_imp = row_imp / (row_imp.mean() + 1e-8)
        col_imp = col_imp / (col_imp.mean() + 1e-8)

        importance[name] = (row_imp, col_imp)

    return importance


def iws_sinkhorn_log(matrix, order=8, row_importance=None, col_importance=None, alpha=0.5,
                     clip_min=1e-3, clip_max=1e3, eps=1e-6, stop_on_increasing_imbalance=True):
    """
    Importance-Weighted Sinkhorn normalization.

    Key modification: Target std varies per row/col based on importance.
    Important rows/cols get lower target std -> tighter normalization -> less quantization error.
    """
    dtype = torch.float32
    m = matrix.to(dtype)
    dev = m.device
    measure = torch.std

    def imbalance(mat):
        s1, s2 = measure(mat, 1), measure(mat, 0)
        s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
        s_max = torch.maximum(s1.max(), s2.max())
        return s_max / s_min

    imb_min = torch.tensor(float('inf'), dtype=dtype, device=dev)
    gate = torch.tensor(0.0, dtype=dtype, device=dev)

    # Base target (same as original SINQ)
    tgt_small = torch.minimum(
        m.std(1).clamp(clip_min, clip_max).min(),
        m.std(0).clamp(clip_min, clip_max).min()
    ) + eps

    # IWS modification: Per-row and per-column targets
    if row_importance is not None:
        # Higher importance -> lower target (1 + alpha*imp) in denominator
        # Clamp importance to avoid extreme targets
        row_imp = row_importance.to(dev).to(dtype).clamp(0.1, 10.0)
        tgt_row = tgt_small / (1 + alpha * row_imp)  # [nrows]
        tgt_row = tgt_row[:, None]  # [nrows, 1] for broadcasting
    else:
        tgt_row = tgt_small

    if col_importance is not None:
        col_imp = col_importance.to(dev).to(dtype).clamp(0.1, 10.0)
        tgt_col = tgt_small / (1 + alpha * col_imp)  # [ncols]
    else:
        tgt_col = tgt_small

    log_mu1 = torch.zeros(m.shape[1], dtype=dtype, device=dev)
    log_mu2 = torch.zeros(m.shape[0], 1, dtype=dtype, device=dev)

    cur0 = m
    ib0 = imbalance(cur0)
    imb_min = torch.minimum(imb_min, ib0)
    mu1_star = log_mu1.exp().clone()
    mu2_star = log_mu2.exp().clone()

    for _ in range(order):
        cur = (m / log_mu1.exp()) / log_mu2.exp()
        ib = imbalance(cur)

        better = (ib <= imb_min).to(dtype)
        imb_min = torch.min(imb_min, ib)
        mu1_star = torch.where(better.bool(), log_mu1.exp(), mu1_star)
        mu2_star = torch.where(better.bool(), log_mu2.exp(), mu2_star)

        if stop_on_increasing_imbalance:
            rising = (ib > imb_min).to(dtype)
            gate = torch.clip(gate + rising, max=1.0)

        g = 1.0 - gate

        std_r = measure(cur, 1).clamp(clip_min, clip_max)
        std_c = measure(cur, 0).clamp(clip_min, clip_max)

        # IWS: Use per-row/col targets instead of uniform tgt_small
        sal_col = (std_c / tgt_col).clamp(0.7, 2.0).log()
        sal_row = (std_r[:, None] / tgt_row).clamp(0.7, 2.0).log()

        log_mu1 = (log_mu1 + (sal_col * g)).clip(-.3, 10.)
        log_mu2 = (log_mu2 + (sal_row * g)).clip(-.3, 10.)

    scaled = m / mu1_star / mu2_star
    return scaled, mu1_star, mu2_star


def quantize_with_iws(model_name, tokenizer, weight_importance, alpha=0.5, device='cuda:0'):
    """
    Quantize a model using IWS-SINQ with importance-weighted Sinkhorn.
    """
    import sinq.dual_shift as dual_shift_module

    original_quantize = dual_shift_module.quantize_dual_scale_shift

    # Track current layer for importance lookup
    current_layer_name = [None]
    layer_counter = [0]

    def patched_quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
        """Patched version with IWS Sinkhorn."""
        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        # Get importance for current layer if available
        layer_name = current_layer_name[0]
        row_imp = None
        col_imp = None

        if layer_name and layer_name in weight_importance:
            row_imp, col_imp = weight_importance[layer_name]
            # Handle shape mismatch (tiles may be smaller than full weight)
            if row_imp.shape[0] != matrix.shape[0]:
                row_imp = None
            if col_imp.shape[0] != matrix.shape[1]:
                col_imp = None

        # Use IWS Sinkhorn
        matrix_normalized, mu1, mu2 = iws_sinkhorn_log(
            matrix, order=16,
            row_importance=row_imp,
            col_importance=col_imp,
            alpha=alpha
        )

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

        return model

    finally:
        dual_shift_module.quantize_dual_scale_shift = original_quantize


def quantize_standard(model_name, tokenizer, device='cuda:0'):
    """Quantize using standard SINQ (baseline)."""
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


def main():
    print("=" * 70)
    print("IWS-SINQ: Importance-Weighted Sinkhorn Test")
    print("=" * 70)
    print("\nHypothesis: Modifying Sinkhorn's target std by importance")
    print("reduces quantization error where it matters most.\n")

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Compute weight importance
    print("=" * 70)
    print("Step 1: Computing per-weight importance via gradients")
    print("=" * 70)

    # Load model for importance computation
    model_for_imp = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)

    # Get calibration texts
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    cal_texts = [s["text"] for s in dataset if len(s["text"].strip()) > 100][:20]

    weight_importance = compute_weight_importance(model_for_imp, tokenizer, cal_texts, device=device)

    print(f"Computed importance for {len(weight_importance)} weight matrices")

    # Show sample importance distribution
    for name in list(weight_importance.keys())[:3]:
        row_imp, col_imp = weight_importance[name]
        print(f"  {name}: row_imp range [{row_imp.min():.3f}, {row_imp.max():.3f}], "
              f"col_imp range [{col_imp.min():.3f}, {col_imp.max():.3f}]")

    del model_for_imp
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: FP16 baseline
    print("\n" + "=" * 70)
    print("Step 2: FP16 Baseline")
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

    # Step 3: Standard SINQ
    print("\n" + "=" * 70)
    print("Step 3: Standard SINQ (baseline)")
    print("=" * 70)

    model_standard = quantize_standard(model_name, tokenizer, device=device)
    model_standard.eval()
    if not hasattr(model_standard, 'hf_device_map'):
        model_standard = model_standard.to(device)
    standard_ppl = evaluate_ppl(model_standard, tokenizer)
    print(f"Standard SINQ PPL: {standard_ppl:.2f}")
    del model_standard
    gc.collect()
    torch.cuda.empty_cache()

    # Step 4: IWS-SINQ with different alpha values
    print("\n" + "=" * 70)
    print("Step 4: IWS-SINQ (importance-weighted Sinkhorn)")
    print("=" * 70)

    alphas_to_test = [0.25, 0.5, 1.0]
    iws_results = {}

    for alpha in alphas_to_test:
        print(f"\nTesting alpha = {alpha}")

        model_iws = quantize_with_iws(model_name, tokenizer, weight_importance, alpha=alpha, device=device)
        model_iws.eval()
        if not hasattr(model_iws, 'hf_device_map'):
            model_iws = model_iws.to(device)

        iws_ppl = evaluate_ppl(model_iws, tokenizer)
        iws_results[alpha] = iws_ppl
        print(f"  IWS-SINQ (α={alpha}): PPL = {iws_ppl:.2f}")

        del model_iws
        gc.collect()
        torch.cuda.empty_cache()

    # Results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Configuration':<30}{'PPL':<12}{'Δ from Standard':<15}")
    print("-" * 60)
    print(f"{'FP16 Baseline':<30}{baseline_ppl:<12.2f}{baseline_ppl - standard_ppl:+.2f}")
    print(f"{'Standard SINQ':<30}{standard_ppl:<12.2f}{0:+.2f}")

    for alpha, ppl in iws_results.items():
        print(f"{'IWS-SINQ (α=' + str(alpha) + ')':<30}{ppl:<12.2f}{ppl - standard_ppl:+.2f}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best_alpha = min(iws_results, key=iws_results.get)
    best_ppl = iws_results[best_alpha]

    if best_ppl < standard_ppl - 0.1:
        print(f"\n✓ SUCCESS: IWS-SINQ (α={best_alpha}) improves PPL by {standard_ppl - best_ppl:.2f}")
        print("  Importance-weighted Sinkhorn targets help!")
    elif abs(best_ppl - standard_ppl) <= 0.1:
        print(f"\n≈ EQUIVALENT: IWS-SINQ ≈ Standard SINQ (best Δ = {best_ppl - standard_ppl:.2f})")
        print("  Importance weighting doesn't help but doesn't hurt.")
    else:
        print(f"\n✗ FAILURE: IWS-SINQ is worse by {best_ppl - standard_ppl:.2f} PPL")
        print("  Importance-weighted targets hurt performance.")
        print("  Root cause: Sinkhorn's coupling may prevent achieving different targets,")
        print("  or the importance weights are not aligned with quantization sensitivity.")


if __name__ == "__main__":
    main()
