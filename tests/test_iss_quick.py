"""
Hypothesis 36: Iterative Sinkhorn-Sparse (ISS)

Core insight: Sinkhorn iterations reveal which weights are "hard to balance."
Weights with high residual across iterations resist normalization.
These weights may be prunable without hurting the final balanced representation.

Mathematical formulation:
- Track per-weight "balancing residual" across Sinkhorn iterations
- Residual = deviation from row/col mean after normalization step
- High cumulative residual → weight is "Sinkhorn-unstable"
- Prune unstable weights → cleaner convergence

This is NOT importance-based pruning. It's convergence-based.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch


def sinkhorn_with_tracking(W, order=16):
    """Sinkhorn normalization that tracks per-weight convergence behavior."""
    K, N = W.shape
    W = W.abs() + 1e-10

    # Track cumulative residual per weight
    cumulative_residual = torch.zeros_like(W)

    mu1 = torch.ones(N, device=W.device, dtype=W.dtype)
    mu2 = torch.ones(K, 1, device=W.device, dtype=W.dtype)

    for iteration in range(order):
        # Column normalization
        col_sums = (W * mu2).sum(dim=0, keepdim=True)
        mu1_new = 1.0 / (col_sums.squeeze() + 1e-10)

        # Row normalization
        row_sums = (W * mu1_new).sum(dim=1, keepdim=True)
        mu2_new = 1.0 / (row_sums + 1e-10)

        # Compute normalized W at this iteration
        W_normalized = W * mu1_new * mu2_new

        # Compute residual: how far each weight is from "balanced" value
        # A perfectly balanced matrix would have equal row/col sums
        row_mean = W_normalized.mean(dim=1, keepdim=True)
        col_mean = W_normalized.mean(dim=0, keepdim=True)

        # Residual = deviation from local mean (row and col)
        residual = (W_normalized - row_mean).abs() + (W_normalized - col_mean).abs()

        # Accumulate residual (later iterations weighted more - closer to convergence)
        weight = (iteration + 1) / order  # Later iterations matter more
        cumulative_residual += weight * residual

        mu1 = mu1_new
        mu2 = mu2_new

    W_final = W * mu1 * mu2

    return W_final, mu1, mu2, cumulative_residual


def test_iss():
    print("="*70)
    print("ITERATIVE SINKHORN-SPARSE (ISS) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    from sinq.sinkhorn import sinkhorn_log
    from sinq.dual_shift import quantize_rtn

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    test_layers = [
        ('gate_proj', model.model.layers[0].mlp.gate_proj.weight.data),
        ('q_proj', model.model.layers[0].self_attn.q_proj.weight.data),
        ('k_proj', model.model.layers[0].self_attn.k_proj.weight.data),
    ]

    sparsity = 0.35
    nbits = 4
    group_size = 64

    for name, W_orig in test_layers:
        print(f"\n{'='*70}")
        print(f"Layer: {name}")
        print("="*70)

        W = W_orig.float()
        K, N = W.shape
        n_groups = N // group_size
        min_max = [0, 2**nbits - 1]
        n_prune = int(K * N * sparsity)

        batch = 64
        X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
        Y_ref = X @ W.T

        # Standard Sinkhorn for baseline
        W_norm_std, mu1_std, mu2_std = sinkhorn_log(W, order=16)
        mu2_std = mu2_std.squeeze()

        # ISS: Sinkhorn with residual tracking
        W_norm_iss, mu1_iss, mu2_iss, residual = sinkhorn_with_tracking(W, order=16)
        mu2_iss = mu2_iss.squeeze()

        # Analyze residual distribution
        print(f"Residual stats: mean={residual.mean():.4f}, std={residual.std():.4f}")
        print(f"Residual range: [{residual.min():.4f}, {residual.max():.4f}]")

        # Standard importance (baseline)
        act_norms = torch.norm(X, dim=0)
        importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1_std.unsqueeze(0) * mu2_std.unsqueeze(1) + 1e-6)

        # ISS importance: Inverse of residual (low residual → high importance)
        # Weights that converge well should be kept
        importance_iss = 1.0 / (residual + 1e-6)

        # Combined: Standard importance weighted by convergence
        importance_combined = importance_std * importance_iss

        # Alternative: Just use residual (high residual → prune)
        # This means low importance = high residual
        importance_residual_based = 1.0 / (residual + 1e-6)

        def evaluate_mask(importance, label):
            threshold = importance.view(-1).sort().values[n_prune]
            mask = (importance > threshold).float()

            W_sparse = W_norm_std * mask
            Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
            Q_g = Q.view(K, n_groups, group_size)
            W_deq = (Q_g - z) * s
            W_deq = W_deq.view(K, N) * mu2_std.unsqueeze(1) * mu1_std.unsqueeze(0)
            Y_approx = X @ W_deq.T
            mse = ((Y_ref - Y_approx) ** 2).mean().item()

            # Compute overlap with standard mask
            threshold_std = importance_std.view(-1).sort().values[n_prune]
            mask_std = (importance_std > threshold_std).float()
            overlap = (mask * mask_std).sum() / mask_std.sum()

            return mse, overlap.item()

        mse_std, _ = evaluate_mask(importance_std, "Standard")
        mse_iss, overlap_iss = evaluate_mask(importance_iss, "ISS (residual)")
        mse_combined, overlap_combined = evaluate_mask(importance_combined, "Combined")

        print(f"\n--- Results ---")
        print(f"Standard (inverse μ): MSE = {mse_std:.6f}")
        improvement_iss = (mse_std - mse_iss) / mse_std * 100
        print(f"ISS (residual-based): MSE = {mse_iss:.6f} ({improvement_iss:+.2f}%), overlap={overlap_iss:.1%}")
        improvement_combined = (mse_std - mse_combined) / mse_std * 100
        print(f"Combined:             MSE = {mse_combined:.6f} ({improvement_combined:+.2f}%), overlap={overlap_combined:.1%}")

        # Also test: Prune HIGH residual weights (direct interpretation)
        importance_prune_unstable = -residual  # Negative so high residual = low importance
        mse_prune_unstable, overlap_unstable = evaluate_mask(importance_prune_unstable, "Prune unstable")
        improvement_unstable = (mse_std - mse_prune_unstable) / mse_std * 100
        print(f"Prune unstable:       MSE = {mse_prune_unstable:.6f} ({improvement_unstable:+.2f}%), overlap={overlap_unstable:.1%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_iss()
