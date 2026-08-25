#!/usr/bin/env python3
"""
Analyze weight outlier distribution for GPU-efficient multi-precision design.

Key questions:
1. Are outliers concentrated in specific COLUMNS (input features)?
2. Are outliers concentrated in specific ROWS (output features)?
3. Are outliers randomly distributed?
4. What fraction of rows/columns contain X% of outliers?

This analysis determines which GPU-efficient representation is most promising:
- Column-concentrated outliers → MicroMix/Atom style column reordering
- Row-concentrated outliers → Row reordering approach
- Block-concentrated outliers → Block-wise precision
- Random → May need different approach
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, '/workspace/SINQ')

# Seed for reproducibility
import random
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def analyze_layer_outliers(weight, layer_name, threshold_percentile=99):
    """
    Analyze outlier distribution in a single weight matrix.

    Returns dict with:
    - row_concentration: fraction of rows containing X% of outliers
    - col_concentration: fraction of cols containing X% of outliers
    - outlier_stats: basic statistics
    """
    W = weight.detach().float().cpu().numpy()

    # Define outliers as values above threshold_percentile
    threshold = np.percentile(np.abs(W), threshold_percentile)
    outlier_mask = np.abs(W) > threshold

    n_rows, n_cols = W.shape
    n_outliers = outlier_mask.sum()

    # Count outliers per row and per column
    outliers_per_row = outlier_mask.sum(axis=1)  # shape: (n_rows,)
    outliers_per_col = outlier_mask.sum(axis=0)  # shape: (n_cols,)

    # Row concentration: What fraction of rows contain 90% of outliers?
    sorted_row_outliers = np.sort(outliers_per_row)[::-1]
    cumsum_row = np.cumsum(sorted_row_outliers)
    target_90pct = 0.9 * n_outliers
    rows_for_90pct = np.searchsorted(cumsum_row, target_90pct) + 1
    row_concentration_90 = rows_for_90pct / n_rows

    # Column concentration: What fraction of columns contain 90% of outliers?
    sorted_col_outliers = np.sort(outliers_per_col)[::-1]
    cumsum_col = np.cumsum(sorted_col_outliers)
    cols_for_90pct = np.searchsorted(cumsum_col, target_90pct) + 1
    col_concentration_90 = cols_for_90pct / n_cols

    # Gini coefficient for row/column concentration
    def gini(values):
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        cumsum = np.cumsum(sorted_vals)
        return (2 * np.sum((np.arange(1, n+1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)

    row_gini = gini(outliers_per_row)
    col_gini = gini(outliers_per_col)

    # Max outlier analysis
    max_outlier = np.abs(W).max()
    mean_outlier = np.abs(W[outlier_mask]).mean() if n_outliers > 0 else 0

    return {
        'layer_name': layer_name,
        'shape': W.shape,
        'n_outliers': n_outliers,
        'threshold': threshold,
        'row_concentration_90': row_concentration_90,  # fraction of rows for 90% outliers
        'col_concentration_90': col_concentration_90,  # fraction of cols for 90% outliers
        'row_gini': row_gini,  # higher = more concentrated
        'col_gini': col_gini,  # higher = more concentrated
        'max_outlier': max_outlier,
        'mean_outlier': mean_outlier,
        'outliers_per_row_std': outliers_per_row.std(),
        'outliers_per_col_std': outliers_per_col.std(),
    }


def analyze_block_outliers(weight, block_size=128):
    """
    Analyze if outliers are concentrated in specific blocks.
    """
    W = weight.detach().float().cpu().numpy()
    n_rows, n_cols = W.shape

    threshold = np.percentile(np.abs(W), 99)
    outlier_mask = np.abs(W) > threshold

    # Count outliers per block
    n_row_blocks = (n_rows + block_size - 1) // block_size
    n_col_blocks = (n_cols + block_size - 1) // block_size

    block_outliers = np.zeros((n_row_blocks, n_col_blocks))

    for i in range(n_row_blocks):
        for j in range(n_col_blocks):
            r_start = i * block_size
            r_end = min(r_start + block_size, n_rows)
            c_start = j * block_size
            c_end = min(c_start + block_size, n_cols)

            block_outliers[i, j] = outlier_mask[r_start:r_end, c_start:c_end].sum()

    total_outliers = outlier_mask.sum()
    n_blocks = n_row_blocks * n_col_blocks

    # What fraction of blocks contain 90% of outliers?
    sorted_block_outliers = np.sort(block_outliers.flatten())[::-1]
    cumsum_block = np.cumsum(sorted_block_outliers)
    target_90pct = 0.9 * total_outliers
    blocks_for_90pct = np.searchsorted(cumsum_block, target_90pct) + 1
    block_concentration_90 = blocks_for_90pct / n_blocks

    return {
        'n_blocks': n_blocks,
        'block_concentration_90': block_concentration_90,
        'max_block_outliers': block_outliers.max(),
        'mean_block_outliers': block_outliers.mean(),
    }


def main():
    print("=" * 70)
    print("OUTLIER DISTRIBUTION ANALYSIS FOR GPU-EFFICIENT MULTI-PRECISION")
    print("=" * 70)

    model_name = "Qwen/Qwen3-1.7B"

    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"  # Load to CPU for analysis
    )

    # Analyze all linear layers
    results = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if module.weight is not None:
                analysis = analyze_layer_outliers(module.weight, name)
                block_analysis = analyze_block_outliers(module.weight)
                analysis.update(block_analysis)
                results.append(analysis)

    print(f"\nAnalyzed {len(results)} linear layers")

    # Aggregate results by layer type
    layer_types = {}
    for r in results:
        # Extract layer type (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
        name = r['layer_name']
        if 'q_proj' in name:
            lt = 'q_proj'
        elif 'k_proj' in name:
            lt = 'k_proj'
        elif 'v_proj' in name:
            lt = 'v_proj'
        elif 'o_proj' in name:
            lt = 'o_proj'
        elif 'gate_proj' in name:
            lt = 'gate_proj'
        elif 'up_proj' in name:
            lt = 'up_proj'
        elif 'down_proj' in name:
            lt = 'down_proj'
        else:
            lt = 'other'

        if lt not in layer_types:
            layer_types[lt] = []
        layer_types[lt].append(r)

    # Print summary by layer type
    print("\n" + "=" * 70)
    print("CONCENTRATION ANALYSIS BY LAYER TYPE")
    print("=" * 70)
    print("\n(Lower concentration = more spread out, Higher = more concentrated)")
    print("Row conc: fraction of rows needed for 90% of outliers")
    print("Col conc: fraction of cols needed for 90% of outliers")
    print("Block conc: fraction of 128×128 blocks needed for 90% of outliers")

    print("\n{:<12} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "Layer Type", "Row Conc", "Col Conc", "Block Conc", "Row Gini", "Col Gini"
    ))
    print("-" * 62)

    for lt in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'other']:
        if lt in layer_types:
            layers = layer_types[lt]
            avg_row_conc = np.mean([l['row_concentration_90'] for l in layers])
            avg_col_conc = np.mean([l['col_concentration_90'] for l in layers])
            avg_block_conc = np.mean([l['block_concentration_90'] for l in layers])
            avg_row_gini = np.mean([l['row_gini'] for l in layers])
            avg_col_gini = np.mean([l['col_gini'] for l in layers])

            print("{:<12} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                lt, avg_row_conc, avg_col_conc, avg_block_conc, avg_row_gini, avg_col_gini
            ))

    # Overall averages
    print("-" * 62)
    avg_row_conc = np.mean([r['row_concentration_90'] for r in results])
    avg_col_conc = np.mean([r['col_concentration_90'] for r in results])
    avg_block_conc = np.mean([r['block_concentration_90'] for r in results])
    avg_row_gini = np.mean([r['row_gini'] for r in results])
    avg_col_gini = np.mean([r['col_gini'] for r in results])

    print("{:<12} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
        "OVERALL", avg_row_conc, avg_col_conc, avg_block_conc, avg_row_gini, avg_col_gini
    ))

    # Key insight extraction
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    if avg_col_conc < avg_row_conc:
        print("\n✓ COLUMNS are more concentrated than rows")
        print(f"  Only {avg_col_conc*100:.1f}% of columns needed for 90% of outliers")
        print("  → Column-wise precision (MicroMix/Atom style) is promising!")
    else:
        print("\n✓ ROWS are more concentrated than columns")
        print(f"  Only {avg_row_conc*100:.1f}% of rows needed for 90% of outliers")
        print("  → Row-wise precision may be more effective")

    if avg_block_conc < min(avg_row_conc, avg_col_conc):
        print(f"\n✓ BLOCKS (128×128) are even more concentrated!")
        print(f"  Only {avg_block_conc*100:.1f}% of blocks needed for 90% of outliers")
        print("  → Block-wise precision could be most efficient")

    # Detailed layer-by-layer analysis
    print("\n" + "=" * 70)
    print("DETAILED LAYER ANALYSIS (First 10 layers)")
    print("=" * 70)

    for r in results[:10]:
        print(f"\n{r['layer_name']}")
        print(f"  Shape: {r['shape']}")
        print(f"  Outliers: {r['n_outliers']} (threshold: {r['threshold']:.4f})")
        print(f"  Row concentration: {r['row_concentration_90']:.3f} (Gini: {r['row_gini']:.3f})")
        print(f"  Col concentration: {r['col_concentration_90']:.3f} (Gini: {r['col_gini']:.3f})")
        print(f"  Block concentration: {r['block_concentration_90']:.3f}")

    # Save results for further analysis
    import json
    output_path = '/workspace/SINQ/results/outlier_distribution.json'
    with open(output_path, 'w') as f:
        # Convert numpy types to Python types
        results_json = []
        for r in results:
            r_json = {}
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    r_json[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    r_json[k] = float(v)
                elif isinstance(v, (np.int32, np.int64)):
                    r_json[k] = int(v)
                else:
                    r_json[k] = v
            results_json.append(r_json)
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
