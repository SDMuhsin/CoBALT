"""
Quick test: Sinkhorn-Structure-Preserving Pruning (SSPP)

NEW OBJECTIVE: Preserve Sinkhorn balance, not just minimize output error.

Current: Prune to minimize ||Y - Y'||²
SSPP: Prune to minimize change in μ factors (preserve variance balance)

Intuition: Sinkhorn finds optimal variance balance. Preserving this balance
after pruning should maintain quantization quality.

Mathematical formulation:
- After pruning weight w[i,j], the effective row/col variance changes
- This changes the "ideal" μ factors
- SSPP minimizes this deviation

Implementation:
- For each weight, compute how much pruning it changes the row/col variance
- Prefer pruning weights that minimally disturb variance balance
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_sspp():
    print("="*70)
    print("SINKHORN-STRUCTURE-PRESERVING PRUNING (SSPP) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    W = model.model.layers[0].mlp.gate_proj.weight.data.float()
    K, N = W.shape
    print(f"Weight shape: [{K}x{N}]")

    # Create activations
    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Current row/col variance
    row_var = W_norm.var(dim=1)  # [K]
    col_var = W_norm.var(dim=0)  # [N]

    # Activation norms
    act_norms = torch.norm(X, dim=0)

    # Standard importance (inverse μ)
    importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # SSPP: Compute variance contribution of each weight
    # Removing w[i,j] changes row_var[i] and col_var[j]
    # Variance = E[(x - μ)²] ≈ E[x²] when centered

    # Contribution to variance: W_norm[i,j]² contributes to row_var[i] and col_var[j]
    W_norm_sq = W_norm ** 2

    # Relative contribution to row variance
    row_var_contrib = W_norm_sq / (row_var.unsqueeze(1) * N + 1e-8)  # [K, N]
    # Relative contribution to col variance
    col_var_contrib = W_norm_sq / (col_var.unsqueeze(0) * K + 1e-8)  # [K, N]

    # Total variance disturbance from removing weight
    var_disturb = row_var_contrib + col_var_contrib  # [K, N]

    # SSPP importance: High if removing would disturb balance
    # Low var_disturb → can safely prune
    # High var_disturb → important for balance
    importance_sspp = importance_std * (1 + var_disturb)

    # Inverse: Penalize high-disturbance weights
    importance_sspp_inv = importance_std / (1 + var_disturb + 1e-6)

    # Alternative: Pure variance-based (ignore standard importance)
    importance_var_only = var_disturb

    # Create masks
    sparsity = 0.35
    n_prune = int(K * N * sparsity)

    def create_mask(importance):
        threshold = importance.view(-1).sort().values[n_prune]
        return (importance > threshold).float()

    mask_std = create_mask(importance_std)
    mask_sspp = create_mask(importance_sspp)
    mask_sspp_inv = create_mask(importance_sspp_inv)
    mask_var = create_mask(importance_var_only)

    # Evaluate
    nbits = 4
    group_size = 64
    min_max = [0, 2**nbits - 1]
    n_groups = N // group_size
    Y_ref = X @ W.T

    def evaluate_mask(mask, name):
        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - z) * s
        W_deq = W_deq_norm.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)

        # Output MSE
        Y_approx = X @ W_deq.T
        out_mse = ((Y_ref - Y_approx) ** 2).mean().item()

        # Variance preservation
        W_sparse_full = W * mask
        W_sparse_norm = W_sparse_full / (mu2.unsqueeze(1) * mu1.unsqueeze(0) + 1e-6)
        # Compute variance of non-zero elements per row/col
        nonzero_mask = mask > 0

        return out_mse

    out_std = evaluate_mask(mask_std, "Standard")
    out_sspp = evaluate_mask(mask_sspp, "SSPP")
    out_sspp_inv = evaluate_mask(mask_sspp_inv, "SSPP-inv")
    out_var = evaluate_mask(mask_var, "Var-only")

    print(f"\n--- Output MSE Results ---")
    print(f"Standard (inverse μ):      {out_std:.6f}")
    print(f"SSPP (preserve balance):   {out_sspp:.6f} ({(out_std-out_sspp)/out_std*100:+.2f}%)")
    print(f"SSPP-inv (allow disturb):  {out_sspp_inv:.6f} ({(out_std-out_sspp_inv)/out_std*100:+.2f}%)")
    print(f"Variance-only:             {out_var:.6f} ({(out_std-out_var)/out_std*100:+.2f}%)")

    # Analyze variance disturbance statistics
    print(f"\n--- Variance Disturbance Statistics ---")
    print(f"Mean: {var_disturb.mean():.6f}")
    print(f"Std:  {var_disturb.std():.6f}")
    print(f"CV:   {var_disturb.std() / var_disturb.mean():.4f}")

    # Mask overlap
    overlap_sspp = (mask_std * mask_sspp).sum() / mask_std.sum()
    overlap_inv = (mask_std * mask_sspp_inv).sum() / mask_std.sum()
    overlap_var = (mask_std * mask_var).sum() / mask_std.sum()

    print(f"\n--- Mask Overlap with Standard ---")
    print(f"SSPP: {overlap_sspp:.1%}")
    print(f"SSPP-inv: {overlap_inv:.1%}")
    print(f"Var-only: {overlap_var:.1%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_sspp()
