"""
Phase 1.2 (New Direction): Analyze SINQ's outlier distribution.

Key Question: How much of SINQ's quantization error comes from a small % of weights?

If top 1% of weights cause >30% of error, outlier isolation could help.
If error is uniformly distributed, outlier isolation won't help.

This validates whether the SpQR-style approach is viable for SINQ.
"""

import os
import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from sinq.sinkhorn import sinkhorn_log
from sinq import dual_shift as ds


def analyze_tile_outliers(W, nbits=3, group_size=128, order=16):
    """
    Analyze outlier distribution for a single weight matrix.

    Returns:
        - error_percentiles: What % of total error comes from top X% of elements
        - outlier_stats: Statistics about outlier positions
    """
    H, D = W.shape
    n_groups = D // group_size

    all_errors = []
    total_mse = 0
    total_elements = 0

    for g in range(n_groups):
        # Extract tile
        tile = W[:, g*group_size:(g+1)*group_size].float()

        # Apply Sinkhorn normalization (same as SINQ)
        tile_norm, mu1, mu2 = sinkhorn_log(tile, order=order)

        # Quantize tile
        min_max = (0, 2**nbits - 1)
        q, scales, z, _ = ds.quantize_rtn(tile_norm, min_max, mode="uniform")

        # Dequantize
        tile_dq = (q.float() - z.float()) * scales.float()

        # Compute per-element error
        errors = (tile_norm - tile_dq) ** 2
        all_errors.append(errors.flatten())
        total_mse += errors.sum().item()
        total_elements += errors.numel()

    # Concatenate all errors
    all_errors = torch.cat(all_errors)

    # Compute percentiles
    sorted_errors, _ = torch.sort(all_errors, descending=True)
    cumsum = torch.cumsum(sorted_errors, dim=0)
    total_error = cumsum[-1].item()

    # What % of total error comes from top X% of elements?
    percentiles = {}
    for pct in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        n_elements = int(len(sorted_errors) * pct / 100)
        if n_elements == 0:
            n_elements = 1
        error_from_top = cumsum[n_elements-1].item()
        percentiles[pct] = error_from_top / total_error * 100

    return {
        'percentiles': percentiles,
        'total_mse': total_mse,
        'total_elements': total_elements,
        'mean_error': total_mse / total_elements,
        'max_error': sorted_errors[0].item(),
        'median_error': sorted_errors[len(sorted_errors)//2].item(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", type=str, default="cpu")  # CPU for analysis
    parser.add_argument("--nbits", type=int, default=3)
    args = parser.parse_args()

    print("="*70)
    print("SINQ OUTLIER ANALYSIS")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Bits: {args.nbits}")

    # Load model weights
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # FP32 for analysis
        device_map='cpu'
    )

    # Analyze each linear layer
    results = []
    layer_names = []

    print("\nAnalyzing layers...")

    for name, param in tqdm(model.named_parameters()):
        if 'weight' not in name:
            continue
        if len(param.shape) != 2:
            continue
        if param.shape[0] < 128 or param.shape[1] < 128:
            continue

        W = param.data.clone()
        stats = analyze_tile_outliers(W, nbits=args.nbits)
        results.append(stats)
        layer_names.append(name)

    print("\n" + "="*70)
    print("OUTLIER DISTRIBUTION ACROSS ALL LAYERS")
    print("="*70)

    # Aggregate statistics
    print("\nWhat % of total error comes from top X% of weights?")
    print("-"*50)

    for pct in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        avg_error_pct = np.mean([r['percentiles'][pct] for r in results])
        min_error_pct = np.min([r['percentiles'][pct] for r in results])
        max_error_pct = np.max([r['percentiles'][pct] for r in results])
        print(f"Top {pct:>4.1f}% of weights → {avg_error_pct:>5.1f}% of error (range: {min_error_pct:.1f}-{max_error_pct:.1f}%)")

    # Key insight
    top_1_pct = np.mean([r['percentiles'][1.0] for r in results])

    print("\n" + "="*70)
    print("KEY INSIGHT FOR OUTLIER HANDLING")
    print("="*70)

    if top_1_pct > 30:
        print(f"""
Top 1% of weights contribute {top_1_pct:.1f}% of total error.

This VALIDATES the outlier handling approach!
- Storing 1% of weights at FP16 would add ~4% memory overhead
- But could eliminate {top_1_pct:.1f}% of quantization error
- SpQR-style approach is PROMISING for SINQ
""")
    elif top_1_pct > 15:
        print(f"""
Top 1% of weights contribute {top_1_pct:.1f}% of total error.

This suggests MODERATE potential for outlier handling.
- Error is somewhat concentrated but not extreme
- May need to isolate 2-5% of weights for meaningful improvement
- Approach is worth testing but with higher memory overhead
""")
    else:
        print(f"""
Top 1% of weights contribute only {top_1_pct:.1f}% of total error.

Error is UNIFORMLY distributed. Outlier handling will NOT help.
- Need to explore other directions
- Consider learned codebooks instead
""")

    # Per-layer breakdown
    print("\n" + "="*70)
    print("PER-LAYER STATISTICS (Top 1% contribution)")
    print("="*70)

    layer_data = [(name, r['percentiles'][1.0], r['mean_error'])
                  for name, r in zip(layer_names, results)]
    layer_data.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Layer':<60} {'Top 1%→Error%':>15}")
    print("-"*75)

    for name, pct, mse in layer_data[:15]:  # Top 15 layers
        short_name = name if len(name) < 60 else "..." + name[-57:]
        print(f"{short_name:<60} {pct:>14.1f}%")

    if len(layer_data) > 15:
        print(f"... and {len(layer_data) - 15} more layers")

    # Error magnitude analysis
    print("\n" + "="*70)
    print("ERROR MAGNITUDE ANALYSIS")
    print("="*70)

    max_errors = [r['max_error'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    median_errors = [r['median_error'] for r in results]

    print(f"Max error across all layers:    {np.mean(max_errors):.6f} (range: {np.min(max_errors):.6f}-{np.max(max_errors):.6f})")
    print(f"Mean error across all layers:   {np.mean(mean_errors):.8f}")
    print(f"Median error across all layers: {np.mean(median_errors):.8f}")
    print(f"Max/Mean ratio:                 {np.mean(max_errors)/np.mean(mean_errors):.1f}x")
    print(f"Max/Median ratio:               {np.mean(max_errors)/np.mean(median_errors):.1f}x")


if __name__ == "__main__":
    main()
