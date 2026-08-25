"""
Debug MWC - verify compensation is actually working
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_mwc_debug():
    print("="*70)
    print("MWC DEBUG - Verify compensation is working")
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
    mu2 = mu2.squeeze()

    # Importance and mask
    act_norms = torch.norm(X, dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)
    threshold = importance.view(-1).sort().values[n_prune]
    mask = (importance > threshold).float()

    W_sparse = W_norm * mask

    # Hessian
    H = X.T @ X  # [N, N]
    H_diag = H.diag()  # [N]

    print(f"\n--- Hessian Analysis ---")
    print(f"H diagonal: mean={H_diag.mean():.4f}, std={H_diag.std():.4f}")
    print(f"H off-diagonal: mean={H.mean():.6f}, max off-diag={H.fill_diagonal_(0).abs().max():.4f}")

    # Compensation
    W_pruned_contrib = W_sparse * (1 - mask)
    print(f"\n--- Pruned Weights ---")
    print(f"Pruned contribution: mean={W_pruned_contrib.mean():.6f}, std={W_pruned_contrib.std():.6f}")
    print(f"Non-zero pruned: {(W_pruned_contrib != 0).sum().item()}")

    # Wait - W_sparse already has zeros where mask=0!
    # So W_pruned_contrib = W_sparse * (1-mask) = 0 everywhere!
    # This is the bug!

    print(f"\n*** BUG FOUND ***")
    print(f"W_sparse already has zeros where pruned, so W_pruned_contrib = 0!")

    # Fix: Use W_norm (before masking) for pruned contribution
    W_pruned_contrib_fixed = W_norm * (1 - mask)
    print(f"\n--- Fixed Pruned Weights (from W_norm) ---")
    print(f"Pruned contribution: mean={W_pruned_contrib_fixed.mean():.6f}, std={W_pruned_contrib_fixed.std():.6f}")
    print(f"Non-zero pruned: {(W_pruned_contrib_fixed != 0).sum().item()}")

    # Compute compensation with fix
    H = X.T @ X
    compensation = (W_pruned_contrib_fixed @ H) / (H_diag.unsqueeze(0) + 1e-8)
    print(f"\n--- Compensation ---")
    print(f"Compensation: mean={compensation.mean():.6f}, std={compensation.std():.6f}")
    print(f"Compensation range: [{compensation.min():.6f}, {compensation.max():.6f}]")

    # Apply compensation
    W_comp = W_sparse - compensation * mask

    # Check difference
    diff = (W_comp - W_sparse).abs()
    print(f"\n--- Difference from sparse ---")
    print(f"Diff: mean={diff.mean():.6f}, max={diff.max():.6f}")
    print(f"Changed weights: {(diff > 1e-8).sum().item()}")

    # Evaluate
    def evaluate(W_comp, label):
        Q, s, z, _ = quantize_rtn(W_comp, min_max, group_size=group_size)
        Q_g = Q.view(K, n_groups, group_size)
        W_deq = (Q_g - z) * s
        W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        Y_approx = X @ W_deq.T
        mse = ((Y_ref - Y_approx) ** 2).mean().item()
        return mse

    mse_sparse = evaluate(W_sparse, "Sparse")
    mse_comp = evaluate(W_comp, "Compensated")
    improvement = (mse_sparse - mse_comp) / mse_sparse * 100
    print(f"\n--- Results ---")
    print(f"Sparse MSE: {mse_sparse:.6f}")
    print(f"Compensated MSE: {mse_comp:.6f} ({improvement:+.2f}%)")

    # Now test MWC with the fix
    print(f"\n--- MWC with fix ---")
    W_pruned_mu = W_pruned_contrib_fixed * mu1.unsqueeze(0)
    compensation_mwc = (W_pruned_mu @ H) / ((mu1 * H_diag).unsqueeze(0) + 1e-8)
    W_comp_mwc = W_sparse - compensation_mwc * mask

    mse_mwc = evaluate(W_comp_mwc, "MWC")
    improvement_mwc = (mse_sparse - mse_mwc) / mse_sparse * 100
    improvement_vs_std = (mse_comp - mse_mwc) / mse_comp * 100
    print(f"MWC MSE: {mse_mwc:.6f} ({improvement_mwc:+.2f}% vs sparse, {improvement_vs_std:+.2f}% vs std)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_mwc_debug()
