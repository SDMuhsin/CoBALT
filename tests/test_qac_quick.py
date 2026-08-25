"""
Quick test: Quantization-Aware Compensation (QAC)

Current: OBS compensates for pruning error, then we quantize.
The compensation assumes perfect weight storage, but quantization adds error.

QAC: After initial prune+compensate+quantize, measure actual error and
apply a second round of compensation.

Idea: Iteratively refine remaining weights to minimize post-quantization error.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_qac():
    print("="*70)
    print("QUANTIZATION-AWARE COMPENSATION (QAC) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    W = model.model.layers[0].mlp.gate_proj.weight.data.float()
    K, N = W.shape
    print(f"Weight shape: [{K}x{N}]")

    sparsity = 0.35
    nbits = 4
    group_size = 64
    n_groups = N // group_size
    min_max = [0, 2**nbits - 1]
    n_prune = int(K * N * sparsity)

    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
    Y_ref = X @ W.T

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2_flat = mu2.squeeze()

    # Importance and mask
    act_norms = torch.norm(X, dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2_flat.unsqueeze(1) + 1e-6)
    threshold = importance.view(-1).sort().values[n_prune]
    mask = (importance > threshold).float()

    # === Standard approach (no QAC) ===
    W_sparse = W_norm * mask
    Q_std, s_std, z_std, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
    Q_g_std = Q_std.view(K, n_groups, group_size)
    W_deq_std = (Q_g_std - z_std) * s_std
    W_deq_std = W_deq_std.view(K, N) * mu2_flat.unsqueeze(1) * mu1.unsqueeze(0)
    Y_std = X @ W_deq_std.T
    mse_std = ((Y_ref - Y_std) ** 2).mean().item()
    print(f"\nStandard MSE: {mse_std:.6f}")

    # === QAC: Iterative correction ===
    print("\n--- QAC: Iterative Correction ---")

    W_current = W_norm * mask

    for iteration in range(3):
        # Quantize
        Q, s, z, _ = quantize_rtn(W_current, min_max, group_size=group_size)
        Q_g = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_g - z) * s
        W_deq = W_deq_norm.view(K, N) * mu2_flat.unsqueeze(1) * mu1.unsqueeze(0)

        # Compute error
        Y_approx = X @ W_deq.T
        error = Y_ref - Y_approx

        mse = ((error) ** 2).mean().item()

        # Correction: Add a small adjustment to remaining weights
        # Optimal adjustment to minimize ||Y_ref - X(W + ΔW)||
        # ΔW = (X^T X)^{-1} X^T error in ideal case

        # Simple approximation: scale weights proportional to per-column activation
        if iteration < 2:  # Only correct for first few iterations
            # Per-column error contribution
            col_error = (X.T @ error).T / (X.shape[0] + 1e-8)  # [K, N] or similar

            # Only adjust kept weights
            # Very conservative adjustment (small step size)
            step_size = 0.1 / (iteration + 1)

            # Approximate: adjust W_norm to reduce error
            # W_norm_new = W_norm - step * (X^T @ error / ||X||^2) / mu_scaling
            # This is tricky because we need to work in normalized space

            # Simpler: adjust the quantization offset slightly
            # z_new = z + delta to shift quantized values

            # Even simpler: just measure error and report
            pass

        print(f"Iter {iteration}: MSE = {mse:.6f}")

    # === Alternative QAC: Adjust scales post-hoc ===
    print("\n--- QAC v2: Scale Adjustment ---")

    # After quantization, find optimal scale per group to minimize output error
    W_sparse = W_norm * mask
    Q, s_orig, z_orig, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
    Q_g = Q.view(K, n_groups, group_size)

    # Original reconstruction
    W_deq_norm = (Q_g - z_orig) * s_orig
    W_deq = W_deq_norm.view(K, N) * mu2_flat.unsqueeze(1) * mu1.unsqueeze(0)
    Y_orig = X @ W_deq.T
    mse_orig = ((Y_ref - Y_orig) ** 2).mean().item()

    # Adjust: find scale that minimizes error for each group
    # For each group, optimal scale = <Y_ref, Y_approx> / <Y_approx, Y_approx>
    # But groups interact, so this is approximate

    s_adjusted = s_orig.clone()

    for g in range(min(5, n_groups)):  # Just test first 5 groups
        # Vary scale for group g
        best_mse = mse_orig
        best_factor = 1.0

        for factor in [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]:
            s_test = s_orig.clone()
            s_test[:, g, :] = s_orig[:, g, :] * factor

            W_deq_test = (Q_g - z_orig) * s_test
            W_deq_test = W_deq_test.view(K, N) * mu2_flat.unsqueeze(1) * mu1.unsqueeze(0)
            Y_test = X @ W_deq_test.T
            mse_test = ((Y_ref - Y_test) ** 2).mean().item()

            if mse_test < best_mse:
                best_mse = mse_test
                best_factor = factor

        if best_factor != 1.0:
            print(f"  Group {g}: best factor = {best_factor:.2f}, MSE {mse_orig:.6f} → {best_mse:.6f}")
            s_adjusted[:, g, :] = s_orig[:, g, :] * best_factor

    # Final with adjusted scales
    W_deq_adj = (Q_g - z_orig) * s_adjusted
    W_deq_adj = W_deq_adj.view(K, N) * mu2_flat.unsqueeze(1) * mu1.unsqueeze(0)
    Y_adj = X @ W_deq_adj.T
    mse_adj = ((Y_ref - Y_adj) ** 2).mean().item()

    improvement = (mse_orig - mse_adj) / mse_orig * 100
    print(f"\nOriginal MSE: {mse_orig:.6f}")
    print(f"Adjusted MSE: {mse_adj:.6f} ({improvement:+.2f}%)")

    # === QAC v3: Zero-point adjustment ===
    print("\n--- QAC v3: Zero-point Adjustment ---")

    best_mse_z = mse_orig
    for z_shift in [-0.5, -0.25, 0, 0.25, 0.5]:
        z_test = z_orig + z_shift

        W_deq_test = (Q_g - z_test) * s_orig
        W_deq_test = W_deq_test.view(K, N) * mu2_flat.unsqueeze(1) * mu1.unsqueeze(0)
        Y_test = X @ W_deq_test.T
        mse_test = ((Y_ref - Y_test) ** 2).mean().item()

        if mse_test < best_mse_z:
            print(f"  z_shift = {z_shift:+.2f}: MSE = {mse_test:.6f}")
            best_mse_z = mse_test

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_qac()
