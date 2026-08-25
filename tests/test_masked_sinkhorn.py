"""
Test: Masked Sinkhorn vs Standard Sinkhorn for Sparse Matrices

Hypothesis: Computing Sinkhorn μ factors only on non-zero entries after pruning
might produce better normalization for quantizing the remaining weights.

SFS failed because zeros corrupt std computation. What if we mask them out?
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import numpy as np


def sinkhorn_log(matrix, order=8, eps=1e-6):
    """Standard SINQ Sinkhorn (std-based)."""
    dtype = torch.float32
    m = matrix.to(dtype)

    clip_min, clip_max = 1e-3, 1e3

    tgt_small = torch.minimum(
        m.std(1).clamp(clip_min, clip_max).min(),
        m.std(0).clamp(clip_min, clip_max).min()
    ) + eps

    log_mu1 = torch.zeros(m.shape[1], dtype=dtype, device=m.device)
    log_mu2 = torch.zeros(m.shape[0], 1, dtype=dtype, device=m.device)

    for _ in range(order):
        cur = (m / log_mu1.exp()) / log_mu2.exp()

        std_r = cur.std(1).clamp(clip_min, clip_max)
        std_c = cur.std(0).clamp(clip_min, clip_max)

        sal_col = (std_c / tgt_small).clamp(0.7, 2.0).log()
        sal_row = (std_r[:, None] / tgt_small).clamp(0.7, 2.0).log()

        log_mu1 = (log_mu1 + sal_col).clip(-.3, 10.)
        log_mu2 = (log_mu2 + sal_row).clip(-.3, 10.)

    mu1 = log_mu1.exp()
    mu2 = log_mu2.exp()
    scaled = m / mu1 / mu2

    return scaled, mu1, mu2.squeeze()


def masked_sinkhorn(matrix, mask, order=8, eps=1e-6):
    """
    Sinkhorn computed only on non-zero (masked) entries.

    For each row/column, compute std only over non-zero entries.
    """
    dtype = torch.float32
    m = matrix.to(dtype)
    mask_f = mask.to(dtype)

    clip_min, clip_max = 1e-3, 1e3

    # Compute initial masked std
    def masked_std(mat, mask, dim):
        """Compute std only over non-zero entries."""
        count = mask.sum(dim=dim, keepdim=True).clamp(min=2)
        mean = (mat * mask).sum(dim=dim, keepdim=True) / count
        var = ((mat - mean) ** 2 * mask).sum(dim=dim, keepdim=True) / count
        return var.sqrt().squeeze()

    # Target: minimum of masked stds
    row_std_init = masked_std(m, mask_f, dim=1)
    col_std_init = masked_std(m, mask_f, dim=0)
    tgt_small = torch.minimum(
        row_std_init.clamp(clip_min, clip_max).min(),
        col_std_init.clamp(clip_min, clip_max).min()
    ) + eps

    log_mu1 = torch.zeros(m.shape[1], dtype=dtype, device=m.device)
    log_mu2 = torch.zeros(m.shape[0], 1, dtype=dtype, device=m.device)

    for _ in range(order):
        cur = (m / log_mu1.exp()) / log_mu2.exp()

        # Masked std computation
        std_r = masked_std(cur, mask_f, dim=1).clamp(clip_min, clip_max)
        std_c = masked_std(cur, mask_f, dim=0).clamp(clip_min, clip_max)

        sal_col = (std_c / tgt_small).clamp(0.7, 2.0).log()
        sal_row = (std_r[:, None] / tgt_small).clamp(0.7, 2.0).log()

        log_mu1 = (log_mu1 + sal_col).clip(-.3, 10.)
        log_mu2 = (log_mu2 + sal_row).clip(-.3, 10.)

    mu1 = log_mu1.exp()
    mu2 = log_mu2.exp()
    scaled = m / mu1 / mu2

    return scaled, mu1, mu2.squeeze()


def test_masked_sinkhorn():
    """Compare masked vs standard Sinkhorn for sparse matrices."""
    print("="*70)
    print("MASKED SINKHORN ANALYSIS")
    print("="*70)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create synthetic sparse matrix (35% sparsity)
    K, N = 256, 512
    W = torch.randn(K, N, device=device, dtype=torch.float32)

    # Create importance-based mask (keep top 65%)
    importance = W.abs()
    threshold = importance.view(-1).quantile(0.35)
    mask = (importance > threshold).float()

    print(f"\nMatrix shape: {K}x{N}")
    print(f"Sparsity: {(1 - mask.mean()):.1%}")
    print(f"Non-zeros per row: {mask.sum(dim=1).mean():.0f}")

    # Method 1: Standard Sinkhorn on FULL matrix, then mask
    W_norm_std, mu1_std, mu2_std = sinkhorn_log(W, order=16)
    W_sparse_std = W_norm_std * mask

    # Method 2: Standard Sinkhorn on SPARSE matrix (zeros included in std)
    W_sparse = W * mask
    W_norm_sfs, mu1_sfs, mu2_sfs = sinkhorn_log(W_sparse, order=16)

    # Method 3: Masked Sinkhorn (zeros excluded from std)
    W_norm_msk, mu1_msk, mu2_msk = masked_sinkhorn(W, mask, order=16)
    W_sparse_msk = W_norm_msk * mask

    # Compare μ factors
    print("\n--- μ Factor Comparison ---")
    print(f"Standard (full):  μ₁ range [{mu1_std.min():.4f}, {mu1_std.max():.4f}], mean {mu1_std.mean():.4f}")
    print(f"SFS (with zeros): μ₁ range [{mu1_sfs.min():.4f}, {mu1_sfs.max():.4f}], mean {mu1_sfs.mean():.4f}")
    print(f"Masked:           μ₁ range [{mu1_msk.min():.4f}, {mu1_msk.max():.4f}], mean {mu1_msk.mean():.4f}")

    print(f"\nStandard (full):  μ₂ range [{mu2_std.min():.4f}, {mu2_std.max():.4f}], mean {mu2_std.mean():.4f}")
    print(f"SFS (with zeros): μ₂ range [{mu2_sfs.min():.4f}, {mu2_sfs.max():.4f}], mean {mu2_sfs.mean():.4f}")
    print(f"Masked:           μ₂ range [{mu2_msk.min():.4f}, {mu2_msk.max():.4f}], mean {mu2_msk.mean():.4f}")

    # Quantize and compare reconstruction
    from sinq.dual_shift import quantize_rtn

    min_max = [0, 7]  # 3-bit
    group_size = 64

    # Quantize all three methods
    Q_std, s_std, z_std, _ = quantize_rtn(W_sparse_std, min_max, group_size=group_size)
    Q_sfs, s_sfs, z_sfs, _ = quantize_rtn(W_norm_sfs, min_max, group_size=group_size)
    Q_msk, s_msk, z_msk, _ = quantize_rtn(W_sparse_msk, min_max, group_size=group_size)

    # Dequantize
    def dequant(Q, s, z, K, N, group_size):
        n_groups = s.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq = (Q_grouped - z) * s
        return W_deq.view(K, N)

    W_deq_std = dequant(Q_std, s_std, z_std, K, N, group_size) * mu2_std.unsqueeze(1) * mu1_std.unsqueeze(0)
    W_deq_sfs = dequant(Q_sfs, s_sfs, z_sfs, K, N, group_size) * mu2_sfs.unsqueeze(1) * mu1_sfs.unsqueeze(0)
    W_deq_msk = dequant(Q_msk, s_msk, z_msk, K, N, group_size) * mu2_msk.unsqueeze(1) * mu1_msk.unsqueeze(0)

    # MSE on non-zero entries only (fair comparison)
    W_orig_masked = W * mask
    mse_std = ((W_orig_masked - W_deq_std * mask) ** 2).sum() / mask.sum()
    mse_sfs = ((W_orig_masked - W_deq_sfs * mask) ** 2).sum() / mask.sum()
    mse_msk = ((W_orig_masked - W_deq_msk * mask) ** 2).sum() / mask.sum()

    print("\n--- Reconstruction MSE (non-zero weights only) ---")
    print(f"Standard (full):  {mse_std:.6f}")
    print(f"SFS (with zeros): {mse_sfs:.6f}")
    print(f"Masked:           {mse_msk:.6f}")

    improvement_vs_std = (mse_std - mse_msk) / mse_std * 100
    improvement_vs_sfs = (mse_sfs - mse_msk) / mse_sfs * 100

    print(f"\nMasked vs Standard: {improvement_vs_std:+.2f}% improvement")
    print(f"Masked vs SFS:      {improvement_vs_sfs:+.2f}% improvement")

    # Test on real weights
    print("\n" + "="*70)
    print("REAL WEIGHTS (Qwen-0.5B layer 0 gate_proj)")
    print("="*70)

    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        W_real = model.model.layers[0].mlp.gate_proj.weight.data.float()
        K_r, N_r = W_real.shape
        print(f"Weight shape: [{K_r}x{N_r}]")

        # Create mask (35% sparsity)
        importance_real = W_real.abs()
        threshold_real = importance_real.view(-1).quantile(0.35)
        mask_real = (importance_real > threshold_real).float()

        print(f"Sparsity: {(1 - mask_real.mean()):.1%}")

        # Standard Sinkhorn on full matrix
        W_norm_std_r, mu1_std_r, mu2_std_r = sinkhorn_log(W_real, order=16)
        W_sparse_std_r = W_norm_std_r * mask_real

        # Masked Sinkhorn
        W_norm_msk_r, mu1_msk_r, mu2_msk_r = masked_sinkhorn(W_real, mask_real, order=16)
        W_sparse_msk_r = W_norm_msk_r * mask_real

        print("\n--- μ Factor Comparison (Real Weights) ---")
        print(f"Standard: μ₁ range [{mu1_std_r.min():.4f}, {mu1_std_r.max():.4f}]")
        print(f"Masked:   μ₁ range [{mu1_msk_r.min():.4f}, {mu1_msk_r.max():.4f}]")

        # Difference in μ values
        mu1_diff = (mu1_std_r - mu1_msk_r).abs()
        mu2_diff = (mu2_std_r - mu2_msk_r).abs()

        print(f"\nμ₁ difference: mean={mu1_diff.mean():.6f}, max={mu1_diff.max():.6f}")
        print(f"μ₂ difference: mean={mu2_diff.mean():.6f}, max={mu2_diff.max():.6f}")

        # If the differences are tiny, masked Sinkhorn gives same results
        if mu1_diff.mean() < 0.001 and mu2_diff.mean() < 0.001:
            print("\n>>> μ factors are nearly IDENTICAL - masked Sinkhorn = standard Sinkhorn")
            print(">>> This angle likely won't provide improvement")
        else:
            print("\n>>> μ factors DIFFER significantly - worth investigating further!")

        # Quantize and compare
        Q_std_r, s_std_r, z_std_r, _ = quantize_rtn(W_sparse_std_r, min_max, group_size=64)
        Q_msk_r, s_msk_r, z_msk_r, _ = quantize_rtn(W_sparse_msk_r, min_max, group_size=64)

        def dequant_real(Q, s, z, K, N, group_size):
            n_groups = s.shape[1]
            Q_grouped = Q.view(K, n_groups, group_size)
            W_deq = (Q_grouped - z) * s
            return W_deq.view(K, N)

        W_deq_std_r = dequant_real(Q_std_r, s_std_r, z_std_r, K_r, N_r, 64) * mu2_std_r.unsqueeze(1) * mu1_std_r.unsqueeze(0)
        W_deq_msk_r = dequant_real(Q_msk_r, s_msk_r, z_msk_r, K_r, N_r, 64) * mu2_msk_r.unsqueeze(1) * mu1_msk_r.unsqueeze(0)

        W_orig_masked_r = W_real * mask_real
        mse_std_r = ((W_orig_masked_r - W_deq_std_r * mask_real) ** 2).sum() / mask_real.sum()
        mse_msk_r = ((W_orig_masked_r - W_deq_msk_r * mask_real) ** 2).sum() / mask_real.sum()

        print(f"\n--- Reconstruction MSE (Real Weights) ---")
        print(f"Standard: {mse_std_r:.6f}")
        print(f"Masked:   {mse_msk_r:.6f}")

        improvement_r = (mse_std_r - mse_msk_r) / mse_std_r * 100
        print(f"Improvement: {improvement_r:+.2f}%")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)


if __name__ == '__main__':
    test_masked_sinkhorn()
