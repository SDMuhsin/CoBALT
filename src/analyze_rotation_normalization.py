"""
Phase 1 Analysis: Understanding how Hadamard rotation and Sinkhorn normalization interact.

This script analyzes:
1. Weight statistics before/after each transformation
2. Outlier distribution (kurtosis, max/mean ratio)
3. Variance imbalance across rows/columns
4. Quantization error with different transformation orders

Goal: Identify if rotation and normalization address orthogonal issues.
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/workspace/SINQ')

from transformers import AutoModelForCausalLM, AutoTokenizer
from sinq.sinkhorn import sinkhorn_log


def hadamard_matrix(n):
    """Generate normalized Hadamard matrix of size n x n (n must be power of 2)."""
    if n == 1:
        return torch.tensor([[1.0]])

    H_half = hadamard_matrix(n // 2)
    H = torch.cat([
        torch.cat([H_half, H_half], dim=1),
        torch.cat([H_half, -H_half], dim=1)
    ], dim=0)
    return H


def apply_hadamard_rows(W, block_size=None):
    """Apply Hadamard rotation to rows of W."""
    m, n = W.shape
    if block_size is None:
        # Find largest power of 2 <= m
        block_size = 2 ** int(np.floor(np.log2(m)))

    H = hadamard_matrix(block_size).to(W.device, W.dtype) / np.sqrt(block_size)

    # Apply blockwise if needed
    if m == block_size:
        return H @ W
    else:
        # Apply to first block_size rows, keep rest unchanged
        W_rot = W.clone()
        W_rot[:block_size] = H @ W[:block_size]
        return W_rot


def apply_hadamard_cols(W, block_size=None):
    """Apply Hadamard rotation to columns of W."""
    m, n = W.shape
    if block_size is None:
        block_size = 2 ** int(np.floor(np.log2(n)))

    H = hadamard_matrix(block_size).to(W.device, W.dtype) / np.sqrt(block_size)

    if n == block_size:
        return W @ H.T
    else:
        W_rot = W.clone()
        W_rot[:, :block_size] = W[:, :block_size] @ H.T
        return W_rot


def compute_statistics(W, name=""):
    """Compute key statistics for a weight matrix."""
    W = W.float()

    # Basic stats
    stats = {
        'name': name,
        'shape': tuple(W.shape),
        'mean': W.mean().item(),
        'std': W.std().item(),
        'min': W.min().item(),
        'max': W.max().item(),
        'abs_max': W.abs().max().item(),
    }

    # Outlier metrics
    abs_vals = W.abs()
    stats['max_mean_ratio'] = (abs_vals.max() / abs_vals.mean()).item()

    # Kurtosis (measure of tail heaviness - high kurtosis = more outliers)
    centered = W - W.mean()
    var = (centered ** 2).mean()
    stats['kurtosis'] = ((centered ** 4).mean() / (var ** 2) - 3).item()

    # Row kurtosis (average across rows)
    row_centered = W - W.mean(dim=1, keepdim=True)
    row_var = (row_centered ** 2).mean(dim=1)
    row_kurt = (row_centered ** 4).mean(dim=1) / (row_var ** 2 + 1e-10) - 3
    stats['row_kurtosis_mean'] = row_kurt.mean().item()
    stats['row_kurtosis_max'] = row_kurt.max().item()

    # Column kurtosis
    col_centered = W - W.mean(dim=0, keepdim=True)
    col_var = (col_centered ** 2).mean(dim=0)
    col_kurt = (col_centered ** 4).mean(dim=0) / (col_var ** 2 + 1e-10) - 3
    stats['col_kurtosis_mean'] = col_kurt.mean().item()
    stats['col_kurtosis_max'] = col_kurt.max().item()

    # Variance imbalance (what Sinkhorn optimizes)
    row_std = W.std(dim=1)
    col_std = W.std(dim=0)
    s_min = min(row_std.min(), col_std.min()).clamp_min(1e-10)
    s_max = max(row_std.max(), col_std.max())
    stats['variance_imbalance'] = (s_max / s_min).item()
    stats['row_std_ratio'] = (row_std.max() / row_std.min().clamp_min(1e-10)).item()
    stats['col_std_ratio'] = (col_std.max() / col_std.min().clamp_min(1e-10)).item()

    return stats


def simulate_quantization_error(W, bits=3):
    """Simulate uniform quantization and return MSE."""
    W = W.float()
    levels = 2 ** bits

    # Per-row quantization (like SINQ tiles)
    w_max = W.amax(dim=1, keepdim=True)
    w_min = W.amin(dim=1, keepdim=True)
    scale = (w_max - w_min) / (levels - 1)
    scale = scale.clamp_min(1e-10)

    W_q = torch.round((W - w_min) / scale) * scale + w_min
    mse = ((W - W_q) ** 2).mean().item()

    return mse


def analyze_layer(W, layer_name):
    """Analyze a single weight matrix with all transformation combinations."""
    print(f"\n{'='*70}")
    print(f"Layer: {layer_name}")
    print(f"Shape: {W.shape}")
    print(f"{'='*70}")

    W = W.float()
    results = []

    # 1. Original weights
    stats_orig = compute_statistics(W, "Original")
    mse_orig = simulate_quantization_error(W)
    stats_orig['quant_mse'] = mse_orig
    results.append(stats_orig)

    # 2. Sinkhorn only
    W_sink, mu1, mu2 = sinkhorn_log(W, order=16)
    stats_sink = compute_statistics(W_sink, "Sinkhorn")
    mse_sink = simulate_quantization_error(W_sink)
    stats_sink['quant_mse'] = mse_sink
    results.append(stats_sink)

    # 3. Hadamard rows only (with appropriate block size)
    m, n = W.shape
    block_m = 2 ** int(np.floor(np.log2(m)))
    block_n = 2 ** int(np.floor(np.log2(n)))

    # For row rotation, we need square blocks or full matrix rotation
    # Apply to columns (more common in QuaRot style)
    W_had = apply_hadamard_cols(W, block_n)
    stats_had = compute_statistics(W_had, "Hadamard(cols)")
    mse_had = simulate_quantization_error(W_had)
    stats_had['quant_mse'] = mse_had
    results.append(stats_had)

    # 4. Hadamard THEN Sinkhorn
    W_had_sink, mu1_hs, mu2_hs = sinkhorn_log(W_had, order=16)
    stats_had_sink = compute_statistics(W_had_sink, "Hadamard→Sinkhorn")
    mse_had_sink = simulate_quantization_error(W_had_sink)
    stats_had_sink['quant_mse'] = mse_had_sink
    results.append(stats_had_sink)

    # 5. Sinkhorn THEN Hadamard
    W_sink_had = apply_hadamard_cols(W_sink, block_n)
    stats_sink_had = compute_statistics(W_sink_had, "Sinkhorn→Hadamard")
    mse_sink_had = simulate_quantization_error(W_sink_had)
    stats_sink_had['quant_mse'] = mse_sink_had
    results.append(stats_sink_had)

    # Print comparison table
    print(f"\n{'Transform':<25} {'Kurtosis':>10} {'Var.Imbal':>12} {'Max/Mean':>10} {'Quant MSE':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['kurtosis']:>10.2f} {r['variance_imbalance']:>12.2f} "
              f"{r['max_mean_ratio']:>10.2f} {r['quant_mse']:>12.6f}")

    return results


def main():
    print("Loading Qwen 1.7B model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B",
        torch_dtype=torch.float16,
        device_map="cpu"  # Keep on CPU for analysis
    )

    # Analyze a few representative layers
    layers_to_analyze = [
        ("model.layers.0.self_attn.q_proj.weight", model.model.layers[0].self_attn.q_proj.weight),
        ("model.layers.0.self_attn.k_proj.weight", model.model.layers[0].self_attn.k_proj.weight),
        ("model.layers.0.mlp.gate_proj.weight", model.model.layers[0].mlp.gate_proj.weight),
        ("model.layers.0.mlp.down_proj.weight", model.model.layers[0].mlp.down_proj.weight),
        ("model.layers.14.self_attn.q_proj.weight", model.model.layers[14].self_attn.q_proj.weight),
        ("model.layers.14.mlp.gate_proj.weight", model.model.layers[14].mlp.gate_proj.weight),
        ("model.layers.27.self_attn.q_proj.weight", model.model.layers[27].self_attn.q_proj.weight),
        ("model.layers.27.mlp.down_proj.weight", model.model.layers[27].mlp.down_proj.weight),
    ]

    all_results = {}
    for name, weight in layers_to_analyze:
        results = analyze_layer(weight.data, name)
        all_results[name] = results

    # Summary: Does rotation help on top of Sinkhorn?
    print("\n" + "="*70)
    print("SUMMARY: MSE Improvement Analysis")
    print("="*70)

    print(f"\n{'Layer':<45} {'Sink/Orig':>12} {'H→S/Sink':>12} {'S→H/Sink':>12}")
    print("-" * 85)

    for name, results in all_results.items():
        short_name = name.split('.')[-2] + '.' + name.split('.')[-1].replace('.weight', '')
        layer_idx = name.split('.')[2]
        short_name = f"L{layer_idx}.{short_name}"

        mse_orig = results[0]['quant_mse']
        mse_sink = results[1]['quant_mse']
        mse_had_sink = results[3]['quant_mse']
        mse_sink_had = results[4]['quant_mse']

        # Ratios (lower is better)
        sink_vs_orig = mse_sink / mse_orig
        had_sink_vs_sink = mse_had_sink / mse_sink
        sink_had_vs_sink = mse_sink_had / mse_sink

        print(f"{short_name:<45} {sink_vs_orig:>12.4f} {had_sink_vs_sink:>12.4f} {sink_had_vs_sink:>12.4f}")

    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("- Sink/Orig < 1: Sinkhorn helps (expected)")
    print("- H→S/Sink < 1: Hadamard before Sinkhorn helps beyond Sinkhorn alone")
    print("- S→H/Sink < 1: Hadamard after Sinkhorn helps beyond Sinkhorn alone")
    print("="*70)


if __name__ == "__main__":
    main()
