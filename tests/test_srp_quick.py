"""
Quick test: Sinkhorn Recalibration after Pruning (SRP)

KEY INSIGHT: Current approach uses μ factors computed on FULL matrix.
But pruning changes variance structure - optimal μ for sparse matrix differs!

Current flow:
1. Sinkhorn(W) → μ₁, μ₂
2. Prune → W_sparse
3. Quantize W_sparse with original μ₁, μ₂

SRP flow:
1. Sinkhorn(W) → μ₁, μ₂ (for importance calculation)
2. Prune → W_sparse
3. Sinkhorn(W_sparse) → μ₁', μ₂' (RECALIBRATE)
4. Quantize W_sparse with NEW μ₁', μ₂'

Why this might help:
- After pruning, row/column variances change
- Original μ factors are suboptimal for sparse matrix
- Recalibrated μ factors should give better quantization
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_srp():
    print("="*70)
    print("SINKHORN RECALIBRATION AFTER PRUNING (SRP) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    # Test on multiple layers
    test_layers = [
        ('gate_proj', model.model.layers[0].mlp.gate_proj.weight.data),
        ('up_proj', model.model.layers[0].mlp.up_proj.weight.data),
        ('down_proj', model.model.layers[0].mlp.down_proj.weight.data),
        ('q_proj', model.model.layers[0].self_attn.q_proj.weight.data),
        ('k_proj', model.model.layers[0].self_attn.k_proj.weight.data),
        ('v_proj', model.model.layers[0].self_attn.v_proj.weight.data),
    ]

    sparsity = 0.35
    nbits = 4
    group_size = 64

    results_standard = []
    results_srp = []

    for name, W_orig in test_layers:
        W = W_orig.float()
        K, N = W.shape
        n_groups = N // group_size
        min_max = [0, 2**nbits - 1]

        # Create activations
        batch = 64
        X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
        Y_ref = X @ W.T

        # Initial Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        # Importance (inverse μ)
        act_norms = torch.norm(X, dim=0)
        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # Prune
        n_prune = int(K * N * sparsity)
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()

        # === STANDARD: Use original μ ===
        W_sparse = W_norm * mask
        Q_std, s_std, z_std, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_grouped_std = Q_std.view(K, n_groups, group_size)
        W_deq_norm_std = (Q_grouped_std - z_std) * s_std
        W_deq_std = W_deq_norm_std.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        Y_std = X @ W_deq_std.T
        mse_std = ((Y_ref - Y_std) ** 2).mean().item()

        # === SRP: Recalibrate Sinkhorn on sparse matrix ===
        # First apply mask to original W (not normalized)
        W_masked = W * mask

        # Handle zeros for Sinkhorn (add small epsilon to zeros)
        W_masked_safe = W_masked.clone()
        W_masked_safe[mask == 0] = 1e-10  # Small value for pruned weights

        # Recalibrate Sinkhorn
        try:
            W_norm_new, mu1_new, mu2_new = sinkhorn_log(W_masked_safe, order=16)
            mu2_new = mu2_new.squeeze()

            # Apply mask again to normalized weights
            W_sparse_new = W_norm_new * mask

            # Quantize with new μ
            Q_srp, s_srp, z_srp, _ = quantize_rtn(W_sparse_new, min_max, group_size=group_size)
            Q_grouped_srp = Q_srp.view(K, n_groups, group_size)
            W_deq_norm_srp = (Q_grouped_srp - z_srp) * s_srp
            W_deq_srp = W_deq_norm_srp.view(K, N) * mu2_new.unsqueeze(1) * mu1_new.unsqueeze(0)
            Y_srp = X @ W_deq_srp.T
            mse_srp = ((Y_ref - Y_srp) ** 2).mean().item()
        except Exception as e:
            print(f"  {name}: SRP failed - {e}")
            mse_srp = mse_std

        results_standard.append(mse_std)
        results_srp.append(mse_srp)

        improvement = (mse_std - mse_srp) / mse_std * 100
        print(f"{name:12s}: Standard={mse_std:.6f}, SRP={mse_srp:.6f} ({improvement:+.2f}%)")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    mean_std = sum(results_standard) / len(results_standard)
    mean_srp = sum(results_srp) / len(results_srp)
    improvement = (mean_std - mean_srp) / mean_std * 100
    print(f"Mean Standard MSE: {mean_std:.6f}")
    print(f"Mean SRP MSE:      {mean_srp:.6f}")
    print(f"Average improvement: {improvement:+.2f}%")

    # Also test alternative: Recompute on masked W directly (set zeros to 0)
    print("\n" + "="*70)
    print("ALTERNATIVE: SRP with zeros kept as zeros")
    print("="*70)

    for name, W_orig in test_layers[:2]:  # Just test 2 layers for speed
        W = W_orig.float()
        K, N = W.shape
        n_groups = N // group_size
        min_max = [0, 2**nbits - 1]

        batch = 64
        X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
        Y_ref = X @ W.T

        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        act_norms = torch.norm(X, dim=0)
        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        n_prune = int(K * N * sparsity)
        threshold = importance.view(-1).sort().values[n_prune]
        mask = (importance > threshold).float()

        # Alternative: Don't recalibrate Sinkhorn on sparse
        # Instead: Apply sparse mask to W, then normalize by row/col means
        W_masked = W * mask

        # Compute row/col scaling from non-zero elements only
        row_sums = (W_masked.abs() * mask).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + 1e-8)
        col_sums = (W_masked.abs() * mask).sum(dim=0, keepdim=True) / (mask.sum(dim=0, keepdim=True) + 1e-8)

        # Heuristic scaling
        scale_row = 1.0 / (row_sums + 1e-6)
        scale_col = 1.0 / (col_sums + 1e-6)

        W_scaled = W_masked * scale_row * scale_col
        W_scaled = W_scaled * mask  # Re-apply mask

        # Quantize
        Q_alt, s_alt, z_alt, _ = quantize_rtn(W_scaled, min_max, group_size=group_size)
        Q_grouped_alt = Q_alt.view(K, n_groups, group_size)
        W_deq_norm_alt = (Q_grouped_alt - z_alt) * s_alt

        # Rescale back
        W_deq_alt = W_deq_norm_alt.view(K, N) / (scale_row * scale_col + 1e-8)

        Y_alt = X @ W_deq_alt.T
        mse_alt = ((Y_ref - Y_alt) ** 2).mean().item()

        # Standard baseline
        W_sparse_std = W_norm * mask
        Q_std, s_std, z_std, _ = quantize_rtn(W_sparse_std, min_max, group_size=group_size)
        Q_grouped_std = Q_std.view(K, n_groups, group_size)
        W_deq_norm_std = (Q_grouped_std - z_std) * s_std
        W_deq_std = W_deq_norm_std.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        Y_std = X @ W_deq_std.T
        mse_std = ((Y_ref - Y_std) ** 2).mean().item()

        improvement = (mse_std - mse_alt) / mse_std * 100
        print(f"{name:12s}: Standard={mse_std:.6f}, Alt={mse_alt:.6f} ({improvement:+.2f}%)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_srp()
