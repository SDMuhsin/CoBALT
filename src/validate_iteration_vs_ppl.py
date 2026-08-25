#!/usr/bin/env python3
"""
Phase 1.5: Micro-Validation - Does Sinkhorn iteration count affect PPL?

CRITICAL TEST: If PPL is flat after 8 iterations, QEP-SINQ is DEAD.

Test setup:
- Pick 3 layers: Layer 0 (high importance), Layer 14 (mid), Layer 27 (low)
- For each layer, test Sinkhorn iterations: [1, 4, 8, 16, 32]
- Quantize ONLY that layer with different iteration counts
- Keep rest of model in FP16
- Measure full-model WikiText-2 PPL

Pass condition: PPL decreases with more iterations on high-importance Layer 0
Fail condition: PPL is flat across all iteration counts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import copy
import sys
sys.path.insert(0, '/workspace/SINQ')

from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_dual_scale_shift


def evaluate_ppl_simple(model, tokenizer, max_samples=20, max_length=512):
    """Simple PPL evaluation on WikiText-2."""
    device = next(model.parameters()).device

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [s["text"] for s in dataset if len(s["text"].strip()) > 100][:max_samples]

    all_text = "\n\n".join(texts)
    encodings = tokenizer(all_text, return_tensors="pt", truncation=True, max_length=max_length * max_samples)
    input_ids = encodings.input_ids.to(device)

    nlls = []
    stride = max_length // 2

    with torch.no_grad():
        for begin in range(0, min(input_ids.size(1), max_length * 10), stride):
            end = min(begin + max_length, input_ids.size(1))
            chunk = input_ids[:, begin:end]

            outputs = model(chunk, labels=chunk)
            nll = outputs.loss.item()

            if not np.isnan(nll) and not np.isinf(nll):
                nlls.append(nll)

            if end >= input_ids.size(1):
                break

    return np.exp(np.mean(nlls)) if nlls else float('inf')


def quantize_weight_with_iterations(weight, num_bits=4, group_size=64, sinkhorn_order=8):
    """
    Quantize a weight matrix using SINQ with specified Sinkhorn iterations.

    This is a simplified version that matches the core SINQ logic.
    """
    device = weight.device
    dtype = weight.dtype

    # Work in float32 for stability
    W = weight.float()

    H, W_dim = W.shape

    # For simplicity, quantize per-group (rows)
    # Reshape into groups
    if W_dim % group_size != 0:
        # Pad
        pad_size = group_size - (W_dim % group_size)
        W = F.pad(W, (0, pad_size))
        W_dim = W.shape[1]

    num_groups = W_dim // group_size
    W_grouped = W.view(H, num_groups, group_size)

    # Quantize each group
    W_q_grouped = torch.zeros_like(W_grouped)

    qmax = 2 ** num_bits - 1

    for g in range(num_groups):
        tile = W_grouped[:, g, :]  # (H, group_size)

        # Apply Sinkhorn normalization with specified iterations
        tile_normalized, mu1, mu2 = sinkhorn_log(
            tile,
            order=sinkhorn_order,
            stop_on_increasing_imbalance=True
        )

        # Compute scales
        s1 = mu1.exp()  # Row scales
        s2 = mu2.exp()  # Column scales

        # Quantize normalized tile
        t_min = tile_normalized.min()
        t_max = tile_normalized.max()
        t_range = t_max - t_min

        if t_range > 0:
            scale = t_range / qmax
            zero = t_min
            q = torch.round((tile_normalized - zero) / scale)
            q = q.clamp(0, qmax)
            tile_dequant = q * scale + zero
        else:
            tile_dequant = tile_normalized

        # De-normalize
        tile_reconstructed = tile_dequant * s1 * s2

        W_q_grouped[:, g, :] = tile_reconstructed

    # Reshape back
    W_q = W_q_grouped.view(H, -1)[:, :weight.shape[1]]

    return W_q.to(dtype)


def test_layer_iterations(model, tokenizer, layer_idx, iterations_list=[1, 4, 8, 16, 32]):
    """
    Test different Sinkhorn iteration counts on a single layer.

    Returns dict mapping iteration count to PPL.
    """
    # Get the linear layers in the specified transformer block
    if hasattr(model, 'model'):
        transformer = model.model
    else:
        transformer = model

    if hasattr(transformer, 'layers'):
        layers = transformer.layers
    else:
        raise ValueError("Cannot find layers")

    layer = layers[layer_idx]

    # Find all linear modules in this layer
    linear_modules = {}
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules[name] = module

    print(f"  Found {len(linear_modules)} linear modules in layer {layer_idx}")

    results = {}

    for sinkhorn_iters in iterations_list:
        print(f"    Testing {sinkhorn_iters} iterations...")

        # Create copies of original weights
        original_weights = {}
        for name, module in linear_modules.items():
            original_weights[name] = module.weight.data.clone()

        # Quantize with specified iterations
        with torch.no_grad():
            for name, module in linear_modules.items():
                W_original = module.weight.data
                W_q = quantize_weight_with_iterations(
                    W_original,
                    num_bits=4,
                    group_size=64,
                    sinkhorn_order=sinkhorn_iters
                )
                module.weight.data = W_q

        # Evaluate PPL
        ppl = evaluate_ppl_simple(model, tokenizer, max_samples=15, max_length=256)
        results[sinkhorn_iters] = ppl
        print(f"      PPL = {ppl:.2f}")

        # Restore original weights
        with torch.no_grad():
            for name, module in linear_modules.items():
                module.weight.data = original_weights[name]

    return results


def main():
    print("=" * 70)
    print("Phase 1.5: Micro-Validation - Sinkhorn Iterations vs PPL")
    print("=" * 70)
    print("\nCRITICAL TEST: Does Sinkhorn iteration count affect full-model PPL?")
    print("If PPL is flat across iterations → QEP-SINQ approach is INVALID\n")

    torch.manual_seed(42)

    model_name = "Qwen/Qwen3-1.7B"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # First, get FP16 baseline
    print("\n" + "=" * 70)
    print("BASELINE: FP16 (no quantization)")
    print("=" * 70)

    baseline_ppl = evaluate_ppl_simple(model, tokenizer, max_samples=15, max_length=256)
    print(f"Baseline PPL: {baseline_ppl:.2f}")

    # Test layers
    # Based on importance analysis:
    # - Layer 0: Highest importance (gradient norm = 0.032)
    # - Layer 14: Mid importance (gradient norm = 0.0025)
    # - Layer 27: Lowest importance (gradient norm = 0.00026)

    test_layers = [0, 14, 27]
    iterations_to_test = [1, 2, 4, 8, 16]

    all_results = {}

    for layer_idx in test_layers:
        print("\n" + "=" * 70)
        print(f"LAYER {layer_idx}")
        print("=" * 70)

        results = test_layer_iterations(
            model, tokenizer, layer_idx,
            iterations_list=iterations_to_test
        )
        all_results[layer_idx] = results

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBaseline FP16 PPL: {baseline_ppl:.2f}")
    print(f"\nPPL by Sinkhorn iterations (single layer quantized to 4-bit):")
    print(f"{'Layer':<8}", end="")
    for iters in iterations_to_test:
        print(f"{iters:>8}", end="")
    print()
    print("-" * (8 + 8 * len(iterations_to_test)))

    for layer_idx in test_layers:
        print(f"Layer {layer_idx:<2}", end="")
        for iters in iterations_to_test:
            ppl = all_results[layer_idx][iters]
            print(f"{ppl:>8.2f}", end="")
        print()

    # Analysis: Check if iterations matter
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    for layer_idx in test_layers:
        results = all_results[layer_idx]
        ppls = [results[i] for i in iterations_to_test]

        ppl_range = max(ppls) - min(ppls)
        ppl_cv = np.std(ppls) / np.mean(ppls)

        # Check if more iterations help (lower PPL)
        trend = results[iterations_to_test[-1]] - results[iterations_to_test[0]]

        print(f"\nLayer {layer_idx}:")
        print(f"  PPL range: {ppl_range:.2f}")
        print(f"  PPL CV: {ppl_cv:.3f}")
        print(f"  Trend (first→last): {'+' if trend > 0 else ''}{trend:.2f}")

        if abs(trend) > 0.5:
            if trend < 0:
                print(f"  → More iterations HELP (PPL decreases)")
            else:
                print(f"  → More iterations HURT (PPL increases)")
        else:
            print(f"  → Iterations have MINIMAL effect")

    # Decision
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)

    # Check Layer 0 (high importance) - does it benefit from more iterations?
    layer0_results = all_results[0]
    layer0_trend = layer0_results[iterations_to_test[-1]] - layer0_results[iterations_to_test[0]]
    layer0_range = max(layer0_results.values()) - min(layer0_results.values())

    # Check Layer 27 (low importance) - should be less sensitive
    layer27_results = all_results[27]
    layer27_range = max(layer27_results.values()) - min(layer27_results.values())

    # Pass criteria:
    # 1. Layer 0 should show PPL improvement with more iterations (negative trend)
    # 2. OR Layer 0 should have higher PPL variance than Layer 27

    passes = False

    if layer0_trend < -0.3:
        print("\n✓ PASS: Layer 0 (high importance) benefits from more iterations")
        print(f"  PPL improved by {-layer0_trend:.2f} from 1 to {iterations_to_test[-1]} iterations")
        passes = True
    elif layer0_range > 0.5 and layer0_range > layer27_range:
        print("\n✓ CONDITIONAL PASS: Layer 0 is more sensitive to iterations than Layer 27")
        print(f"  Layer 0 PPL range: {layer0_range:.2f}")
        print(f"  Layer 27 PPL range: {layer27_range:.2f}")
        passes = True
    else:
        print("\n✗ FAIL: Sinkhorn iterations have minimal effect on PPL")
        print(f"  Layer 0 PPL range: {layer0_range:.2f}")
        print(f"  Layer 0 trend: {layer0_trend:.2f}")
        print("\n  QEP-SINQ approach may not work.")
        print("  Consider alternative mechanisms for importance weighting.")

    if passes:
        print("\n  Proceed to Phase 2: Hypothesis Formulation")
    else:
        print("\n  ABORT: Return to Phase 1.1 with new approach")

    return {
        "baseline_ppl": baseline_ppl,
        "results": all_results,
        "passes": passes
    }


if __name__ == "__main__":
    main()
