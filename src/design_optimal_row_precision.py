#!/usr/bin/env python3
"""
Design optimal row-wise precision allocation within bit budget.

Given:
- Base precision: 5 bits (SINQ 5-bit baseline)
- Budget: 5.25 bits average (0.25 extra bits per weight on average)
- Goal: Maximize error reduction

Key insight from analysis:
- 2.70% of rows have >2x average error, contributing 9.13% of error
- 1.11% of rows have >3x average error, contributing 5.34% of error

With 0.25 extra bits budget:
- Can give 1 extra bit to 25% of rows, OR
- Can give 2 extra bits to 12.5% of rows, OR
- Can give 3 extra bits to 8.3% of rows

This script determines:
1. Optimal allocation strategy within budget
2. Expected error reduction
3. GPU-efficient representation
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


def get_row_errors(weight, nbits=5):
    """Get per-row quantization errors."""
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

    return row_errors


def compute_error_with_allocation(row_errors, allocation, extra_bits_per_level):
    """
    Compute total error given an allocation of extra bits.

    allocation: dict mapping extra_bits -> fraction of rows
    extra_bits_per_level: list like [1, 2, 3] meaning levels 0,1,2,3 extra bits

    Error reduction per extra bit: 4x (quadrature reduction)
    """
    total_error = row_errors.sum()
    n_rows = len(row_errors)

    # Sort rows by error (descending)
    sorted_indices = np.argsort(row_errors)[::-1]
    sorted_errors = row_errors[sorted_indices]

    reduced_error = 0.0
    row_idx = 0

    for extra_bits, frac in sorted(allocation.items(), key=lambda x: -x[0]):
        # Give extra_bits to the top frac of remaining rows
        n_rows_this_level = int(frac * n_rows)
        end_idx = min(row_idx + n_rows_this_level, n_rows)

        # Error reduction factor: 4^extra_bits
        reduction_factor = 4 ** extra_bits
        errors_this_level = sorted_errors[row_idx:end_idx]
        reduced_error += errors_this_level.sum() / reduction_factor

        row_idx = end_idx

    # Remaining rows get base precision (no reduction)
    reduced_error += sorted_errors[row_idx:].sum()

    return reduced_error


def find_optimal_allocation(row_errors, bit_budget=0.25, max_extra_bits=4):
    """
    Find optimal allocation of extra bits within budget.

    Uses greedy approach: assign extra bits to rows with highest error first.
    """
    n_rows = len(row_errors)
    total_error = row_errors.sum()

    # Sort rows by error
    sorted_indices = np.argsort(row_errors)[::-1]
    sorted_errors = row_errors[sorted_indices]

    best_reduction = 0
    best_allocation = {}

    # Try different strategies
    strategies = []

    # Strategy 1: Uniform extra bits to top K%
    for extra_bits in range(1, max_extra_bits + 1):
        max_frac = bit_budget / extra_bits
        for frac in np.arange(0.01, min(max_frac + 0.01, 1.0), 0.01):
            strategies.append({extra_bits: frac})

    # Strategy 2: Two-tier allocation
    for eb1 in range(1, max_extra_bits + 1):
        for eb2 in range(1, eb1):
            for f1 in np.arange(0.01, bit_budget / eb1, 0.01):
                remaining_budget = bit_budget - f1 * eb1
                max_f2 = remaining_budget / eb2
                for f2 in np.arange(0.01, min(max_f2, 1.0 - f1), 0.01):
                    if f1 * eb1 + f2 * eb2 <= bit_budget:
                        strategies.append({eb1: f1, eb2: f2})

    print(f"Evaluating {len(strategies)} allocation strategies...")

    for alloc in strategies:
        # Check budget
        used_bits = sum(eb * frac for eb, frac in alloc.items())
        if used_bits > bit_budget + 0.001:
            continue

        reduced_error = compute_error_with_allocation(row_errors, alloc, None)
        reduction = (total_error - reduced_error) / total_error

        if reduction > best_reduction:
            best_reduction = reduction
            best_allocation = alloc.copy()

    return best_allocation, best_reduction


def main():
    print("=" * 70)
    print("OPTIMAL ROW-PRECISION ALLOCATION DESIGN")
    print("=" * 70)
    print("\nGoal: Find best allocation within 5.25 bit budget")
    print("(Base: 5-bit SINQ, Budget: 0.25 extra bits on average)")

    model_name = "Qwen/Qwen3-1.7B"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )

    # Collect row errors from all layers
    all_row_errors = []
    layer_info = []

    print("\nCollecting row errors from all layers...")

    for layer_idx in range(28):
        layer = model.model.layers[layer_idx]

        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(layer.self_attn, proj_name)
            else:
                proj = getattr(layer.mlp, proj_name)

            weight = proj.weight.data.to(device)
            row_errors = get_row_errors(weight, nbits=5)

            all_row_errors.append(row_errors)
            layer_info.append({
                'layer': layer_idx,
                'proj': proj_name,
                'n_rows': len(row_errors),
                'total_error': row_errors.sum(),
                'max_error_ratio': row_errors.max() / row_errors.mean()
            })

    # Concatenate all row errors (treating all layers uniformly)
    combined_errors = np.concatenate(all_row_errors)
    total_error = combined_errors.sum()
    n_total_rows = len(combined_errors)

    print(f"\nTotal rows across all layers: {n_total_rows}")
    print(f"Total error (5-bit): {total_error:.2e}")

    # Find optimal allocation
    print("\n" + "=" * 70)
    print("FINDING OPTIMAL ALLOCATION")
    print("=" * 70)

    best_alloc, best_reduction = find_optimal_allocation(
        combined_errors, bit_budget=0.25, max_extra_bits=4
    )

    print(f"\nBest allocation found:")
    total_bits_used = 0
    for extra_bits, frac in sorted(best_alloc.items(), key=lambda x: -x[0]):
        n_rows = int(frac * n_total_rows)
        bits_used = frac * extra_bits
        total_bits_used += bits_used
        print(f"  +{extra_bits} bits to {frac*100:.2f}% of rows ({n_rows} rows)")
        print(f"    → Uses {bits_used:.4f} average bits")

    print(f"\nTotal average bits: 5 + {total_bits_used:.4f} = {5 + total_bits_used:.4f}")
    print(f"Error reduction: {best_reduction*100:.2f}%")

    # Verify budget
    if 5 + total_bits_used <= 5.25:
        print(f"✓ Within 5.25 bit budget")
    else:
        print(f"✗ Exceeds 5.25 bit budget!")

    # Compare with simple strategies
    print("\n" + "=" * 70)
    print("COMPARISON WITH SIMPLE STRATEGIES")
    print("=" * 70)

    simple_strategies = [
        ({1: 0.25}, "25% rows +1 bit"),
        ({2: 0.125}, "12.5% rows +2 bits"),
        ({3: 0.083}, "8.3% rows +3 bits"),
        ({4: 0.0625}, "6.25% rows +4 bits"),
    ]

    print("\n{:<25} {:>10} {:>15}".format("Strategy", "Avg Bits", "Error Reduction"))
    print("-" * 50)

    for alloc, name in simple_strategies:
        bits_used = sum(eb * frac for eb, frac in alloc.items())
        reduced_error = compute_error_with_allocation(combined_errors, alloc, None)
        reduction = (total_error - reduced_error) / total_error
        print("{:<25} {:>10.4f} {:>14.2f}%".format(name, 5 + bits_used, reduction * 100))

    print("{:<25} {:>10.4f} {:>14.2f}%".format("Optimal", 5 + total_bits_used, best_reduction * 100))

    # GPU efficiency consideration
    print("\n" + "=" * 70)
    print("GPU EFFICIENCY CONSIDERATION")
    print("=" * 70)

    # For row-wise precision to be GPU-efficient, we need:
    # 1. Reorder rows by precision (offline)
    # 2. Process high-precision and low-precision regions separately
    # 3. Sum results

    high_prec_frac = sum(frac for _, frac in best_alloc.items())
    print(f"\nHigh-precision rows: {high_prec_frac*100:.2f}%")
    print(f"Low-precision rows: {(1-high_prec_frac)*100:.2f}%")

    print("\nGPU-efficient approach:")
    print("1. Reorder weight matrix rows: [High-prec rows | Low-prec rows]")
    print("2. Split input activation: X_high = X[:, high_indices], X_low = X[:, low_indices]")
    print("   (Note: For W @ X, rows of W correspond to output features)")
    print("3. Compute: Y = W_high @ X + W_low @ X")
    print("   OR use single kernel with precision boundary")

    print("\n⚠ CHALLENGE: Row-wise precision affects OUTPUT features")
    print("  Unlike column-wise (Atom/MicroMix), can't easily reorder")
    print("  Would need custom kernel handling different precisions")

    print("\n" + "=" * 70)
    print("ALTERNATIVE: COLUMN-WISE ANALYSIS")
    print("=" * 70)

    # Re-analyze with column-wise approach
    all_col_errors = []

    for layer_idx in range(28):
        layer = model.model.layers[layer_idx]

        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            if proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                proj = getattr(layer.self_attn, proj_name)
            else:
                proj = getattr(layer.mlp, proj_name)

            weight = proj.weight.data.to(device)
            W = weight.float()
            n_rows, n_cols = W.shape

            col_errors = np.zeros(n_cols)
            tile_size = 128
            n_col_tiles = (n_cols + tile_size - 1) // tile_size

            for t in range(n_col_tiles):
                c_start = t * tile_size
                c_end = min(c_start + tile_size, n_cols)
                tile = W[:, c_start:c_end]

                min_max = (0, 2**5 - 1)
                q, s1, s2, z = quantize_dual_scale_shift(tile, min_max, method='sinq')
                tile_deq = (q.float() - z) * s1 * s2
                error_sq = (tile.cpu().numpy() - tile_deq.cpu().numpy()) ** 2

                for local_c in range(error_sq.shape[1]):
                    col_errors[c_start + local_c] = error_sq[:, local_c].sum()

            all_col_errors.append(col_errors)

    combined_col_errors = np.concatenate(all_col_errors)
    total_col_error = combined_col_errors.sum()
    n_total_cols = len(combined_col_errors)

    print(f"\nTotal columns across all layers: {n_total_cols}")

    best_col_alloc, best_col_reduction = find_optimal_allocation(
        combined_col_errors, bit_budget=0.25, max_extra_bits=4
    )

    print(f"\nOptimal COLUMN allocation:")
    total_col_bits_used = 0
    for extra_bits, frac in sorted(best_col_alloc.items(), key=lambda x: -x[0]):
        n_cols = int(frac * n_total_cols)
        bits_used = frac * extra_bits
        total_col_bits_used += bits_used
        print(f"  +{extra_bits} bits to {frac*100:.2f}% of columns ({n_cols} cols)")

    print(f"\nAverage bits: {5 + total_col_bits_used:.4f}")
    print(f"Error reduction: {best_col_reduction*100:.2f}%")

    # Final recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if best_reduction > best_col_reduction:
        print(f"\n✓ ROW-wise allocation is more effective ({best_reduction*100:.2f}% vs {best_col_reduction*100:.2f}%)")
        print("  BUT row-wise is harder to make GPU-efficient")
    else:
        print(f"\n✓ COLUMN-wise allocation is comparable ({best_col_reduction*100:.2f}% vs {best_reduction*100:.2f}%)")
        print("  AND column-wise IS GPU-efficient (like Atom/MicroMix)")

    print("\n⚠ Key insight: Error reductions are MODEST (~5-10%)")
    print("  This is because SINQ+Sinkhorn already distributes error well")
    print("  Multi-precision provides incremental improvement, not breakthrough")


if __name__ == "__main__":
    main()
