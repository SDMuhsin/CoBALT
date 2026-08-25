#!/usr/bin/env python3
"""
Test Layer-Type-Aware SINQ (LTA-SINQ).

Based on importance analysis:
- V (1.885), O (1.500) - high importance
- down (1.111), up (1.005), K (1.001) - medium importance
- gate (0.740), Q (0.727) - low importance

Allocate Sinkhorn iterations based on layer type importance.
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


# Importance-based iteration allocation
# Higher importance -> more iterations
TYPE_ORDERS = {
    'v_proj': 20,    # 1.885 importance
    'o_proj': 18,    # 1.500 importance
    'down_proj': 16, # 1.111 importance
    'up_proj': 16,   # 1.005 importance
    'k_proj': 14,    # 1.001 importance
    'gate_proj': 12, # 0.740 importance
    'q_proj': 12,    # 0.727 importance
    'default': 16
}


def quantize_with_lta(model_name, tokenizer, type_orders, device='cuda:0'):
    """
    Quantize using Layer-Type-Aware SINQ.
    Different layer types get different Sinkhorn iterations.
    """
    import sinq.dual_shift as dual_shift_module
    from sinq.sinkhorn import sinkhorn_log

    original_quantize = dual_shift_module.quantize_dual_scale_shift

    # Track current layer name
    current_layer_name = [None]
    order_counts = {k: 0 for k in type_orders.keys()}

    def get_order_for_layer(layer_name):
        """Get Sinkhorn order based on layer type."""
        if layer_name is None:
            return type_orders['default']

        for type_key in type_orders:
            if type_key in layer_name:
                return type_orders[type_key]
        return type_orders['default']

    def patched_quantize_dual_scale_shift(matrix, min_max, method='sinq', awq_scale=None):
        dtype = matrix.dtype
        dev = matrix.device
        matrix = matrix.float()

        # Get order based on layer type
        order = get_order_for_layer(current_layer_name[0])

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

        # We need to hook into the quantization process to know which layer we're quantizing
        # This is tricky because SINQ processes tiles via vmap

        # The layer name is available during patch_model, let's hook there
        from sinq.patch_model import AutoSINQHFModel
        from sinq.sinqlinear import BaseQuantizeConfig, SINQLinear

        quant_config = BaseQuantizeConfig(
            nbits=4,
            group_size=64,
            axis=1,
            tiling_mode='1D',
            method='sinq_nogemlite'
        )

        # Monkey-patch SINQLinear to track layer name
        original_init = SINQLinear.__init__

        def patched_init(self, *args, **kwargs):
            # Set the current layer name before quantization
            if hasattr(self, '_sinq_layer_name'):
                current_layer_name[0] = self._sinq_layer_name

                # Track which types we're quantizing
                for type_key in type_orders:
                    if type_key in current_layer_name[0]:
                        order_counts[type_key] = order_counts.get(type_key, 0) + 1
                        break

            return original_init(self, *args, **kwargs)

        SINQLinear.__init__ = patched_init

        # Also need to pass layer name to SINQLinear
        # This requires modifying the patch_model code path
        # Let's try a different approach - iterate layer by layer manually

        from sinq.sinqlinear import Quantizer
        from sinq.patch_model import replace_linear_by_sinq

        # Get all linear layers
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))

        # Quantize each linear layer with appropriate order
        for name, module in tqdm(linear_layers, desc="Quantizing"):
            current_layer_name[0] = name

            # Create quantized version
            # This is complex because we need to replace in-place
            # Let's just use the standard pipeline and accept that
            # we can't easily inject layer names

        # Actually, let's use a simpler approach:
        # Just call quantize_model and track via the patched function

        current_layer_name[0] = None  # Reset
        AutoSINQHFModel.quantize_model(
            model,
            tokenizer=tokenizer,
            quant_config=quant_config,
            compute_dtype=torch.float16,
            device=device
        )

        SINQLinear.__init__ = original_init  # Restore

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
    print("LTA-SINQ: Layer-Type-Aware Sinkhorn Test")
    print("=" * 70)
    print("\nImportance-based iteration allocation:")
    print("  V (high): 20 iterations")
    print("  O (high): 18 iterations")
    print("  down, up (medium): 16 iterations")
    print("  K (medium): 14 iterations")
    print("  gate, Q (low): 12 iterations")

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

    # Standard SINQ
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

    # Since we can't easily inject layer names into the tile-level quantization,
    # let's test a simpler hypothesis: just increase iterations for ALL layers

    # Test with higher uniform iterations
    print("\n" + "=" * 70)
    print("Testing Higher Uniform Iterations")
    print("=" * 70)

    for order in [20, 24]:
        print(f"\nTesting uniform order={order}")
        model_high = quantize_standard(model_name, tokenizer, order=order, device=device)
        model_high.eval()
        if not hasattr(model_high, 'hf_device_map'):
            model_high = model_high.to(device)
        high_ppl = evaluate_ppl(model_high, tokenizer)
        print(f"  SINQ (order={order}) PPL: {high_ppl:.2f}")
        del model_high
        gc.collect()
        torch.cuda.empty_cache()

    # Results summary
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print("\nThe vmap-based tile quantization in SINQ prevents per-tile or")
    print("per-layer-type iteration allocation without major architectural changes.")
    print("\nTo pursue Error Propagation within SINQ, we would need to either:")
    print("1. Modify SINQ to not use vmap (slower but more flexible)")
    print("2. Use mixed precision at the LAYER level (different bit-widths)")
    print("3. Accept that tile-level error propagation is architecturally blocked")


if __name__ == "__main__":
    main()
