#!/usr/bin/env python3
"""
Test Depth-Sensitive SINQ (DS-SINQ).

Key insight: Sensitivity varies dramatically by layer DEPTH, not layer type.
- Last layers (21-27) are highly sensitive to quantization
- Early layers (0-7) often tolerate or even benefit from quantization
- Gradient importance does NOT correlate with quantization sensitivity

Strategy:
- Final layers (22-27): Higher Sinkhorn iterations (order=24)
- Middle layers (11-21): Standard iterations (order=16)
- Early layers (0-10): Lower iterations (order=8) - they're tolerant

This is within Error Propagation Framework: allocating quantization resources
based on how errors propagate through layer depth.
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


def get_layer_depth(layer_name):
    """Extract layer depth from layer name."""
    import re
    match = re.search(r'layers\.(\d+)', layer_name)
    if match:
        return int(match.group(1))
    return -1  # For non-layer modules (embeddings, etc.)


def get_depth_order(layer_name, num_layers=28):
    """
    Get Sinkhorn order based on layer depth.

    Depth-sensitive allocation:
    - Final 6 layers (22-27): order=24 (most sensitive)
    - Middle layers (11-21): order=16 (standard)
    - Early layers (0-10): order=8 (most tolerant)
    """
    depth = get_layer_depth(layer_name)

    if depth < 0:
        return 16  # Default for non-layer modules

    if depth >= 22:  # Last 6 layers
        return 24
    elif depth >= 11:  # Middle layers
        return 16
    else:  # Early layers (0-10)
        return 8


def quantize_with_ds_sinq(model_name, tokenizer, device='cuda:0'):
    """
    Quantize using Depth-Sensitive SINQ.
    Different depths get different Sinkhorn iterations.
    """
    import sinq.dual_shift as dual_shift_module
    from sinq.sinkhorn import sinkhorn_log

    original_quantize = dual_shift_module.quantize_dual_scale_shift

    # Track current layer and order counts
    current_layer_name = [None]
    order_counts = {8: 0, 16: 0, 24: 0}

    def patched_quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        # Get order based on layer depth
        order = get_depth_order(current_layer_name[0])
        order_counts[order] = order_counts.get(order, 0) + 1

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

        # Hook to track layer name during quantization
        from sinq.sinqlinear import SINQLinear
        original_init = SINQLinear.__init__

        def patched_init(self, *args, **kwargs):
            if hasattr(self, '_sinq_layer_name'):
                current_layer_name[0] = self._sinq_layer_name
            return original_init(self, *args, **kwargs)

        SINQLinear.__init__ = patched_init

        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device=device
        )

        SINQLinear.__init__ = original_init

        print(f"  Order allocation: {order_counts}")
        return model

    finally:
        dual_shift_module.quantize_dual_scale_shift = original_quantize


def quantize_standard(model_name, tokenizer, order=16, device='cuda:0'):
    """Quantize using standard SINQ (uniform order)."""
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
    print("DS-SINQ: Depth-Sensitive Sinkhorn Test")
    print("=" * 70)
    print("\nBased on empirical sensitivity measurement:")
    print("  - Layer 27 down_proj: +11.1 PPL degradation (CRITICAL)")
    print("  - Layer 0 down_proj: -0.59 PPL degradation (HELPS)")
    print("  - Gradient importance does NOT correlate with quant sensitivity")
    print("\nDepth-sensitive allocation:")
    print("  - Final layers (22-27): order=24 (high iterations)")
    print("  - Middle layers (11-21): order=16 (standard)")
    print("  - Early layers (0-10): order=8 (reduced, tolerant)")

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Standard SINQ (uniform order=16)
    print("\n" + "=" * 70)
    print("Standard SINQ (uniform order=16)")
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

    # DS-SINQ
    print("\n" + "=" * 70)
    print("DS-SINQ (Depth-Sensitive)")
    print("=" * 70)

    model_ds = quantize_with_ds_sinq(model_name, tokenizer, device=device)
    model_ds.eval()
    if not hasattr(model_ds, 'hf_device_map'):
        model_ds = model_ds.to(device)
    ds_ppl = evaluate_ppl(model_ds, tokenizer)
    print(f"DS-SINQ PPL: {ds_ppl:.2f}")
    del model_ds
    gc.collect()
    torch.cuda.empty_cache()

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nFP16 Baseline:  {baseline_ppl:.2f}")
    print(f"Standard SINQ:  {standard_ppl:.2f}")
    print(f"DS-SINQ:        {ds_ppl:.2f}")

    improvement = standard_ppl - ds_ppl
    if improvement > 0:
        print(f"\n✓ DS-SINQ improves PPL by {improvement:.2f} ({improvement/standard_ppl*100:.1f}%)")
    else:
        print(f"\n✗ DS-SINQ does not improve PPL ({improvement:.2f})")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    if ds_ppl < standard_ppl:
        print("\nDS-SINQ demonstrates that depth-aware resource allocation")
        print("based on TRUE sensitivity improves quantization quality.")
    else:
        print("\nThe depth-sensitive iteration allocation via patching may")
        print("not be effective because SINQ's vmap-based tile quantization")
        print("doesn't allow the patched function to know the current layer.")
        print("Mixed-precision at the BIT-WIDTH level (not iteration level)")
        print("may be the only viable path within SINQ's architecture.")


if __name__ == "__main__":
    main()
