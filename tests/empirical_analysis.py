"""
Empirical Analysis for SINQ-Sparse Research.

Purpose: Ground hypothesis generation in empirical observations.
This analysis addresses questions raised by vetting agents:
1. How often do pruned weights affect quantization scales?
2. What is the distribution of Sinkhorn factors?
3. What is the correlation between importance and quantization error?
4. What makes some weights "harder" to quantize?
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import numpy as np
from collections import defaultdict
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import compute_activation_norms, create_sparsity_mask


def analyze_scale_changes(W, importance, sparsity, nbits, group_size):
    """
    Q1: How often do pruned weights affect quantization scales?

    For each group, check if the pruned weights include the min or max.
    """
    K, N = W.shape
    n_prune_per_row = int(N * sparsity)

    # Get indices of weights to prune (lowest importance per row)
    _, sorted_idx = importance.sort(dim=1)
    prune_idx = sorted_idx[:, :n_prune_per_row]

    # Create mask
    mask = torch.ones_like(W)
    for i in range(K):
        mask[i, prune_idx[i]] = 0

    # Analyze each group
    n_groups = N // group_size
    boundary_pruned = 0
    total_groups = 0
    scale_change_magnitude = []

    for i in range(K):
        for g in range(n_groups):
            g_start = g * group_size
            g_end = (g + 1) * group_size

            group_weights = W[i, g_start:g_end]
            group_mask = mask[i, g_start:g_end]

            # Original scale
            orig_max = group_weights.max()
            orig_min = group_weights.min()
            orig_scale = (orig_max - orig_min) / (2**nbits - 1)

            # Scale after pruning
            kept_weights = group_weights[group_mask == 1]
            if len(kept_weights) > 0:
                new_max = kept_weights.max()
                new_min = kept_weights.min()
                new_scale = (new_max - new_min) / (2**nbits - 1)
            else:
                new_scale = orig_scale

            # Check if boundary was pruned
            orig_max_idx = group_weights.argmax()
            orig_min_idx = group_weights.argmin()

            max_pruned = (group_mask[orig_max_idx] == 0)
            min_pruned = (group_mask[orig_min_idx] == 0)

            if max_pruned or min_pruned:
                boundary_pruned += 1
                scale_change = abs(new_scale - orig_scale) / (orig_scale + 1e-8)
                scale_change_magnitude.append(scale_change.item())

            total_groups += 1

    boundary_rate = boundary_pruned / total_groups
    avg_scale_change = np.mean(scale_change_magnitude) if scale_change_magnitude else 0

    return {
        'boundary_prune_rate': boundary_rate,
        'avg_scale_change': avg_scale_change,
        'num_boundary_pruned': boundary_pruned,
        'total_groups': total_groups
    }


def analyze_sinkhorn_factors(W):
    """
    Q2: What is the distribution of Sinkhorn factors?
    """
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # mu2 might be [K,1] or [K]
    if mu2.dim() > 1:
        mu2 = mu2.squeeze()

    return {
        'mu1_mean': mu1.mean().item(),
        'mu1_std': mu1.std().item(),
        'mu1_min': mu1.min().item(),
        'mu1_max': mu1.max().item(),
        'mu2_mean': mu2.mean().item(),
        'mu2_std': mu2.std().item(),
        'mu2_min': mu2.min().item(),
        'mu2_max': mu2.max().item(),
        'mu1_cv': (mu1.std() / mu1.mean()).item(),  # coefficient of variation
        'mu2_cv': (mu2.std() / mu2.mean()).item(),
    }


def analyze_importance_quant_correlation(W, X, nbits, group_size):
    """
    Q3: What is the correlation between importance and quantization error?
    """
    K, N = W.shape

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)
    if mu2.dim() > 1:
        mu2 = mu2.squeeze()

    # Importance
    act_norms = compute_activation_norms(X)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2.unsqueeze(1)

    # Quantize
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    # Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0)

    # Per-weight quantization error
    quant_error = (W - W_deq).abs()

    # Flatten for correlation
    imp_flat = importance.flatten().cpu().numpy()
    err_flat = quant_error.flatten().cpu().numpy()

    # Correlation
    correlation = np.corrcoef(imp_flat, err_flat)[0, 1]

    # Also check if high-importance weights have lower error (as we'd hope)
    high_imp_mask = imp_flat > np.median(imp_flat)
    low_imp_mask = ~high_imp_mask

    high_imp_err = err_flat[high_imp_mask].mean()
    low_imp_err = err_flat[low_imp_mask].mean()

    return {
        'importance_error_correlation': correlation,
        'high_importance_mean_error': high_imp_err,
        'low_importance_mean_error': low_imp_err,
        'error_ratio': high_imp_err / (low_imp_err + 1e-8)
    }


def analyze_outliers(W):
    """
    Q4: What makes some weights "harder" to quantize?
    Analyze outlier distribution.
    """
    W_flat = W.flatten()

    # Statistics
    mean = W_flat.mean().item()
    std = W_flat.std().item()

    # Outliers (> 3 std from mean)
    outlier_mask = (W_flat - mean).abs() > 3 * std
    outlier_rate = outlier_mask.float().mean().item()

    # Per-row outlier analysis
    K, N = W.shape
    row_outlier_rates = []
    for i in range(K):
        row = W[i]
        row_mean = row.mean()
        row_std = row.std()
        row_outliers = ((row - row_mean).abs() > 3 * row_std).float().mean().item()
        row_outlier_rates.append(row_outliers)

    # Per-column outlier analysis
    col_outlier_rates = []
    for j in range(N):
        col = W[:, j]
        col_mean = col.mean()
        col_std = col.std()
        col_outliers = ((col - col_mean).abs() > 3 * col_std).float().mean().item()
        col_outlier_rates.append(col_outliers)

    return {
        'global_outlier_rate': outlier_rate,
        'row_outlier_rate_mean': np.mean(row_outlier_rates),
        'row_outlier_rate_std': np.std(row_outlier_rates),
        'col_outlier_rate_mean': np.mean(col_outlier_rates),
        'col_outlier_rate_std': np.std(col_outlier_rates),
    }


def run_analysis_synthetic():
    """Run analysis on synthetic data."""
    print("="*60)
    print("EMPIRICAL ANALYSIS: Synthetic Data")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 256, 512
    batch = 64
    sparsity = 0.35
    nbits = 3
    group_size = 64

    W = torch.randn(K, N, device=device)
    X = torch.randn(batch, N, device=device)

    # Sinkhorn for importance
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)
    if mu2.dim() > 1:
        mu2 = mu2.squeeze()
    act_norms = compute_activation_norms(X)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2.unsqueeze(1)

    print(f"\nMatrix: {K}x{N}, sparsity={sparsity*100:.0f}%, {nbits}-bit, group_size={group_size}")

    # Q1: Scale changes
    print("\n--- Q1: Scale Change Analysis ---")
    scale_results = analyze_scale_changes(W, importance, sparsity, nbits, group_size)
    print(f"Boundary prune rate: {scale_results['boundary_prune_rate']*100:.2f}%")
    print(f"Avg scale change when boundary pruned: {scale_results['avg_scale_change']*100:.2f}%")

    # Q2: Sinkhorn factors
    print("\n--- Q2: Sinkhorn Factor Distribution ---")
    sink_results = analyze_sinkhorn_factors(W)
    print(f"μ₁: mean={sink_results['mu1_mean']:.4f}, std={sink_results['mu1_std']:.4f}, CV={sink_results['mu1_cv']:.4f}")
    print(f"μ₂: mean={sink_results['mu2_mean']:.4f}, std={sink_results['mu2_std']:.4f}, CV={sink_results['mu2_cv']:.4f}")

    # Q3: Importance-error correlation
    print("\n--- Q3: Importance vs Quantization Error ---")
    corr_results = analyze_importance_quant_correlation(W, X, nbits, group_size)
    print(f"Correlation: {corr_results['importance_error_correlation']:.4f}")
    print(f"High-importance mean error: {corr_results['high_importance_mean_error']:.6f}")
    print(f"Low-importance mean error: {corr_results['low_importance_mean_error']:.6f}")

    # Q4: Outliers
    print("\n--- Q4: Outlier Analysis ---")
    outlier_results = analyze_outliers(W)
    print(f"Global outlier rate (>3σ): {outlier_results['global_outlier_rate']*100:.2f}%")

    return {**scale_results, **sink_results, **corr_results, **outlier_results}


def run_analysis_real():
    """Run analysis on real Qwen-0.5B weights."""
    print("\n" + "="*60)
    print("EMPIRICAL ANALYSIS: Qwen-0.5B Real Weights")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM

        print("\nLoading Qwen-0.5B...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        layers_to_analyze = [0, 10, 20]
        projections = ['gate_proj', 'up_proj', 'down_proj']

        all_results = defaultdict(list)

        for layer_idx in layers_to_analyze:
            layer = model.model.layers[layer_idx]

            for proj_name in projections:
                proj = getattr(layer.mlp, proj_name)
                W = proj.weight.data.float()
                K, N = W.shape
                device = W.device

                # Synthetic activations
                batch = 64
                X = torch.randn(batch, N, device=device, dtype=torch.float32)

                sparsity = 0.35
                nbits = 3
                group_size = 64

                # Importance
                W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
                if mu2.dim() > 1:
                    mu2 = mu2.squeeze()
                act_norms = compute_activation_norms(X)
                importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2.unsqueeze(1)

                print(f"\n--- Layer {layer_idx} {proj_name} [{K}x{N}] ---")

                # Q1
                scale_results = analyze_scale_changes(W, importance, sparsity, nbits, group_size)
                print(f"Boundary prune rate: {scale_results['boundary_prune_rate']*100:.2f}%")
                all_results['boundary_prune_rate'].append(scale_results['boundary_prune_rate'])

                # Q2
                sink_results = analyze_sinkhorn_factors(W)
                print(f"μ₁ CV: {sink_results['mu1_cv']:.4f}, μ₂ CV: {sink_results['mu2_cv']:.4f}")
                all_results['mu1_cv'].append(sink_results['mu1_cv'])
                all_results['mu2_cv'].append(sink_results['mu2_cv'])

                # Q3
                corr_results = analyze_importance_quant_correlation(W, X, nbits, group_size)
                print(f"Importance-error correlation: {corr_results['importance_error_correlation']:.4f}")
                all_results['correlation'].append(corr_results['importance_error_correlation'])

                # Q4
                outlier_results = analyze_outliers(W)
                print(f"Outlier rate: {outlier_results['global_outlier_rate']*100:.2f}%")
                all_results['outlier_rate'].append(outlier_results['global_outlier_rate'])

        print("\n" + "="*60)
        print("SUMMARY STATISTICS (across all analyzed layers)")
        print("="*60)
        print(f"Avg boundary prune rate: {np.mean(all_results['boundary_prune_rate'])*100:.2f}% ± {np.std(all_results['boundary_prune_rate'])*100:.2f}%")
        print(f"Avg μ₁ CV: {np.mean(all_results['mu1_cv']):.4f}")
        print(f"Avg μ₂ CV: {np.mean(all_results['mu2_cv']):.4f}")
        print(f"Avg importance-error correlation: {np.mean(all_results['correlation']):.4f}")
        print(f"Avg outlier rate: {np.mean(all_results['outlier_rate'])*100:.2f}%")

        del model
        torch.cuda.empty_cache()

        return all_results

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    run_analysis_synthetic()
    run_analysis_real()
