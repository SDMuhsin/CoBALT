#!/usr/bin/env python3
"""
Analyze EXTREME ERROR rows/columns more carefully.

Key observation from previous analysis:
- Max row error is 8-20x average (some rows have huge error)
- Max col error is only 1.2-2.6x average
- Overall concentration is weak (82-85%)

This suggests: While MOST errors are uniform, there are a FEW extreme outlier rows.
If we can identify and handle these extreme rows with higher precision,
we may get significant error reduction with minimal bit overhead.

Analysis:
1. What fraction of rows have >5x average error?
2. What fraction of total error do these extreme rows contribute?
3. Are these extreme rows consistent across layers?
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM
import sys
sys.path.insert(0, '/workspace/SINQ')

from sinq.dual_shift import quantize_dual_scale_shift

# Seed
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def analyze_extreme_error_rows(weight, nbits=4, thresholds=[2, 3, 5, 10]):
    """
    Analyze extreme error rows.

    Returns analysis of what fraction of rows exceed various thresholds
    and how much error they contribute.
    """
    device = weight.device
    W = weight.float()
    n_rows, n_cols = W.shape

    tile_size = 128
    row_errors = np.zeros(n_rows)

    n_col_tiles = (n_cols + tile_size - 1) // tile_size

    for t in range(n_col_tiles):
        c_start = t * tile_size
        c_end = min(c_start + tile_size, n_cols)
        tile = W[:, c_start:c_end]

        min_max = (0, 2**nbits - 1)
        q, s1, s2, z = quantize_dual_scale_shift(tile, min_max, method='sinq')
        tile_deq = (q.float() - z) * s1 * s2
        error_sq = (tile.cpu().numpy() - tile_deq.cpu().numpy()) ** 2
        row_errors += error_sq.sum(axis=1)

    total_error = row_errors.sum()
    avg_error = row_errors.mean()

    results = {'avg_row_error': avg_error, 'total_error': total_error}

    for thresh in thresholds:
        extreme_mask = row_errors > (thresh * avg_error)
        n_extreme = extreme_mask.sum()
        frac_extreme = n_extreme / n_rows
        error_from_extreme = row_errors[extreme_mask].sum() / total_error

        results[f'rows_>{thresh}x'] = n_extreme
        results[f'frac_rows_>{thresh}x'] = frac_extreme
        results[f'error_from_>{thresh}x'] = error_from_extreme

    # Also analyze what precision boost would help
    # If we give 1 extra bit to extreme rows, error is reduced by 4x
    # Calculate potential error reduction

    results['max_row_error'] = row_errors.max()
    results['max_row_idx'] = row_errors.argmax()
    results['max_row_ratio'] = row_errors.max() / avg_error

    return results, row_errors


def main():
    print("=" * 70)
    print("EXTREME ERROR ROW ANALYSIS")
    print("=" * 70)
    print("\nQuestion: Can we get significant error reduction by giving")
    print("higher precision to just a FEW extreme error rows?\n")

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )

    # Analyze all layers, aggregate extreme rows
    layer_results = []

    print("\nAnalyzing all layers...")

    for layer_idx in range(28):
        layer = model.model.layers[layer_idx]

        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(layer.self_attn, proj_name)
            else:
                proj = getattr(layer.mlp, proj_name)

            weight = proj.weight.data.to(device)
            results, row_errors = analyze_extreme_error_rows(weight, nbits=4)
            results['layer'] = layer_idx
            results['proj'] = proj_name
            results['n_rows'] = weight.shape[0]
            layer_results.append(results)

    print(f"\nAnalyzed {len(layer_results)} weight matrices")

    # Summary statistics
    print("\n" + "=" * 70)
    print("EXTREME ROW ANALYSIS")
    print("=" * 70)

    thresholds = [2, 3, 5, 10]
    print("\n{:>12} {:>12} {:>12} {:>12} {:>12}".format(
        "Threshold", "Avg Rows%", "Avg Error%", "Max Rows%", "Max Error%"
    ))
    print("-" * 60)

    for thresh in thresholds:
        frac_key = f'frac_rows_>{thresh}x'
        error_key = f'error_from_>{thresh}x'

        avg_frac = np.mean([r[frac_key] for r in layer_results]) * 100
        avg_error = np.mean([r[error_key] for r in layer_results]) * 100
        max_frac = np.max([r[frac_key] for r in layer_results]) * 100
        max_error = np.max([r[error_key] for r in layer_results]) * 100

        print("{:>12} {:>11.2f}% {:>11.2f}% {:>11.2f}% {:>11.2f}%".format(
            f">{thresh}x avg", avg_frac, avg_error, max_frac, max_error
        ))

    # Key calculation: Potential benefit
    print("\n" + "=" * 70)
    print("POTENTIAL BENEFIT CALCULATION")
    print("=" * 70)

    # Average: X% of rows contribute Y% of error
    # If we give those rows 1 extra bit, their error reduces by 4x
    # Net error reduction = Y% * 0.75 = 0.75Y%
    # Bit overhead = X% extra bits

    print("\nScenario: Give 1 extra bit to extreme rows")
    print("(1 extra bit reduces quantization error by ~4x)")

    for thresh in thresholds:
        frac_key = f'frac_rows_>{thresh}x'
        error_key = f'error_from_>{thresh}x'

        avg_frac = np.mean([r[frac_key] for r in layer_results])
        avg_error = np.mean([r[error_key] for r in layer_results])

        # 1 extra bit for these rows
        bit_overhead = avg_frac * 1.0  # extra bit per row
        error_reduction = avg_error * 0.75  # 75% of their error eliminated

        print(f"\n>{thresh}x threshold:")
        print(f"  Rows affected: {avg_frac*100:.2f}%")
        print(f"  Error from these rows: {avg_error*100:.2f}%")
        print(f"  Bit overhead: +{bit_overhead*100:.2f}% (e.g., 4.0 → {4*(1+bit_overhead):.3f} bits)")
        print(f"  Error reduction: -{error_reduction*100:.2f}%")
        print(f"  Efficiency: {error_reduction/bit_overhead:.2f}x error reduction per bit")

    # By projection type
    print("\n" + "=" * 70)
    print("BY PROJECTION TYPE (>5x threshold)")
    print("=" * 70)

    proj_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    print("\n{:<12} {:>10} {:>12} {:>10} {:>10}".format(
        "Proj Type", "Rows%", "Error%", "Max Ratio", "Efficiency"
    ))

    for pt in proj_types:
        pt_results = [r for r in layer_results if r['proj'] == pt]
        if pt_results:
            avg_frac = np.mean([r['frac_rows_>5x'] for r in pt_results]) * 100
            avg_error = np.mean([r['error_from_>5x'] for r in pt_results]) * 100
            avg_max_ratio = np.mean([r['max_row_ratio'] for r in pt_results])
            efficiency = (avg_error * 0.75) / (avg_frac * 1.0) if avg_frac > 0 else 0

            print("{:<12} {:>9.2f}% {:>11.2f}% {:>10.1f}x {:>10.2f}x".format(
                pt, avg_frac, avg_error, avg_max_ratio, efficiency
            ))

    # KEY INSIGHT
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    # Check if >5x threshold gives good efficiency
    avg_frac_5x = np.mean([r['frac_rows_>5x'] for r in layer_results])
    avg_error_5x = np.mean([r['error_from_>5x'] for r in layer_results])

    print(f"\nWith >5x threshold:")
    print(f"  {avg_frac_5x*100:.2f}% of rows have >5x average error")
    print(f"  These rows contribute {avg_error_5x*100:.2f}% of total error")

    if avg_error_5x > 2 * avg_frac_5x:
        print(f"\n✓ ERROR IS CONCENTRATED in extreme rows!")
        print(f"  Error contribution ({avg_error_5x*100:.1f}%) >> Row fraction ({avg_frac_5x*100:.1f}%)")
        print(f"  → Giving extra precision to extreme rows is EFFICIENT")

        # Calculate what we can achieve
        base_bits = 5.0
        extra_bits_per_extreme_row = 2.0  # Give 2 extra bits
        bit_overhead = avg_frac_5x * extra_bits_per_extreme_row
        error_reduction = avg_error_5x * 0.9375  # 2 bits = 16x reduction ≈ 94%

        avg_bits = base_bits + bit_overhead
        print(f"\n  Example: 5-bit base + 2 extra bits for extreme rows")
        print(f"  Average bits: {avg_bits:.3f}")
        print(f"  Error reduction: {error_reduction*100:.1f}%")

    else:
        print(f"\n⚠ Error is relatively uniform across rows")
        print(f"  Multi-precision per row may provide limited benefit")


if __name__ == "__main__":
    main()
