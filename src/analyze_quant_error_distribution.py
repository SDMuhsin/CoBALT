#!/usr/bin/env python3
"""
Analyze QUANTIZATION ERROR distribution for GPU-efficient multi-precision design.

This is different from outlier distribution - we care about which rows/columns
contribute most to the QUANTIZATION ERROR, not just which have the largest values.

Key insight: Sinkhorn normalization may redistribute outliers, but some rows/columns
may still contribute disproportionately to error after normalization.

Analysis:
1. Apply SINQ quantization to each layer
2. Measure per-row and per-column contribution to MSE
3. Determine if errors are concentrated or uniformly distributed
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, '/workspace/SINQ')

from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_dual_scale_shift

# Seed for reproducibility
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def analyze_quant_error_distribution(weight, nbits=4, group_size=64):
    """
    Analyze how quantization error is distributed across rows/columns.

    Returns:
    - row_error: error per row (shape: n_rows)
    - col_error: error per column (shape: n_cols)
    - concentration metrics
    """
    device = weight.device
    dtype = weight.dtype
    W = weight.float()

    # SINQ uses 1D tiling: tiles of shape [H × 128]
    n_rows, n_cols = W.shape

    # For simplicity, analyze on the full matrix with group_size=128 (one tile column)
    # This mimics SINQ's approach

    tile_size = 128
    row_errors = np.zeros(n_rows)
    col_errors = np.zeros(n_cols)

    n_col_tiles = (n_cols + tile_size - 1) // tile_size

    for t in range(n_col_tiles):
        c_start = t * tile_size
        c_end = min(c_start + tile_size, n_cols)

        tile = W[:, c_start:c_end]

        # Apply Sinkhorn normalization
        min_max = (0, 2**nbits - 1)
        q, s1, s2, z = quantize_dual_scale_shift(tile, min_max, method='sinq')

        # Dequantize
        tile_deq = (q.float() - z) * s1 * s2

        # Error per element
        error_sq = (tile.cpu().numpy() - tile_deq.cpu().numpy()) ** 2

        # Accumulate row errors
        row_errors += error_sq.sum(axis=1)

        # Accumulate column errors
        for local_c in range(error_sq.shape[1]):
            global_c = c_start + local_c
            col_errors[global_c] = error_sq[:, local_c].sum()

    total_error = row_errors.sum()

    # Concentration analysis
    def concentration_for_90pct(values):
        sorted_vals = np.sort(values)[::-1]
        cumsum = np.cumsum(sorted_vals)
        target = 0.9 * cumsum[-1]
        idx = np.searchsorted(cumsum, target) + 1
        return idx / len(values)

    def gini(values):
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)

    row_conc_90 = concentration_for_90pct(row_errors)
    col_conc_90 = concentration_for_90pct(col_errors)
    row_gini = gini(row_errors)
    col_gini = gini(col_errors)

    return {
        'total_mse': total_error / (n_rows * n_cols),
        'row_conc_90': row_conc_90,
        'col_conc_90': col_conc_90,
        'row_gini': row_gini,
        'col_gini': col_gini,
        'row_errors': row_errors,
        'col_errors': col_errors,
        'max_row_error_ratio': row_errors.max() / (row_errors.mean() + 1e-10),
        'max_col_error_ratio': col_errors.max() / (col_errors.mean() + 1e-10),
    }


def main():
    print("=" * 70)
    print("QUANTIZATION ERROR DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print("\nThis analyzes where SINQ's quantization error comes from.")
    print("If errors are concentrated in specific rows/columns,")
    print("we can allocate more precision to those regions.\n")

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )

    # Analyze subset of layers for speed
    results = []
    layer_indices = [0, 5, 10, 15, 20, 25, 27]  # Sample across depth

    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]

        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(layer.self_attn, proj_name)
            else:
                proj = getattr(layer.mlp, proj_name)

            weight = proj.weight.data.to(device)

            print(f"\nAnalyzing layer {layer_idx} {proj_name} (shape: {tuple(weight.shape)})...")

            analysis = analyze_quant_error_distribution(weight, nbits=4)

            results.append({
                'layer': layer_idx,
                'proj': proj_name,
                'shape': tuple(weight.shape),
                **{k: v for k, v in analysis.items() if not isinstance(v, np.ndarray)}
            })

            print(f"  MSE: {analysis['total_mse']:.6f}")
            print(f"  Row concentration (90%): {analysis['row_conc_90']:.3f} (Gini: {analysis['row_gini']:.3f})")
            print(f"  Col concentration (90%): {analysis['col_conc_90']:.3f} (Gini: {analysis['col_gini']:.3f})")
            print(f"  Max row error ratio: {analysis['max_row_error_ratio']:.2f}x average")
            print(f"  Max col error ratio: {analysis['max_col_error_ratio']:.2f}x average")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    avg_row_conc = np.mean([r['row_conc_90'] for r in results])
    avg_col_conc = np.mean([r['col_conc_90'] for r in results])
    avg_row_gini = np.mean([r['row_gini'] for r in results])
    avg_col_gini = np.mean([r['col_gini'] for r in results])

    print(f"\nAverage row concentration (90%): {avg_row_conc:.3f}")
    print(f"Average col concentration (90%): {avg_col_conc:.3f}")
    print(f"Average row Gini: {avg_row_gini:.3f}")
    print(f"Average col Gini: {avg_col_gini:.3f}")

    # By projection type
    print("\n" + "-" * 40)
    print("BY PROJECTION TYPE")
    print("-" * 40)

    proj_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    print("\n{:<12} {:>10} {:>10} {:>10} {:>10}".format(
        "Proj Type", "Row Conc", "Col Conc", "Max Row", "Max Col"
    ))

    for pt in proj_types:
        pt_results = [r for r in results if r['proj'] == pt]
        if pt_results:
            avg_rc = np.mean([r['row_conc_90'] for r in pt_results])
            avg_cc = np.mean([r['col_conc_90'] for r in pt_results])
            avg_mr = np.mean([r['max_row_error_ratio'] for r in pt_results])
            avg_mc = np.mean([r['max_col_error_ratio'] for r in pt_results])
            print("{:<12} {:>10.3f} {:>10.3f} {:>10.1f}x {:>10.1f}x".format(
                pt, avg_rc, avg_cc, avg_mr, avg_mc
            ))

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT FOR GPU-EFFICIENT MULTI-PRECISION")
    print("=" * 70)

    if avg_row_conc < avg_col_conc:
        print(f"\n✓ ERRORS are more concentrated in ROWS than columns")
        print(f"  Only {avg_row_conc*100:.1f}% of rows account for 90% of error")
        print(f"  → Row-wise precision allocation is promising")
        print(f"  → But ROW-WISE precision is harder to make GPU-efficient!")
    else:
        print(f"\n✓ ERRORS are more concentrated in COLUMNS than rows")
        print(f"  Only {avg_col_conc*100:.1f}% of columns account for 90% of error")
        print(f"  → Column-wise precision (MicroMix/Atom style) is promising!")
        print(f"  → This is GPU-efficient via column reordering!")

    # Check if concentration is strong enough
    if avg_row_conc < 0.5 or avg_col_conc < 0.5:
        print("\n✓ STRONG CONCENTRATION detected!")
        print("  Multi-precision within layers is worthwhile.")
    else:
        print("\n⚠ WEAK CONCENTRATION - errors are relatively uniform")
        print("  Multi-precision may provide limited benefit.")


if __name__ == "__main__":
    main()
