"""
Quick test: Sinkhorn-Guided Semi-Structured Sparsity (SGSS)

KEY INSIGHT: Hardware prefers structured patterns (N:M sparsity).
Can we use Sinkhorn μ factors to guide a semi-structured pattern?

Idea: Different rows/columns get different sparsity levels based on μ.
- High μ rows → can be pruned more (50-60%)
- Low μ rows → prune less (20-30%)

This is "semi-structured": variable density per row, but predictable pattern.

Potential novelty: Bridge between unstructured (best quality) and N:M (best speed).
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_sgss():
    print("="*70)
    print("SINKHORN-GUIDED SEMI-STRUCTURED SPARSITY (SGSS) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    W = model.model.layers[0].mlp.gate_proj.weight.data.float()
    K, N = W.shape
    print(f"Weight shape: [{K}x{N}]")

    target_sparsity = 0.35
    nbits = 4
    group_size = 64
    n_groups = N // group_size
    min_max = [0, 2**nbits - 1]

    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
    Y_ref = X @ W.T

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Importance
    act_norms = torch.norm(X, dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # === Uniform baseline ===
    n_prune = int(K * N * target_sparsity)
    threshold = importance.view(-1).sort().values[n_prune]
    mask_uniform = (importance > threshold).float()

    def evaluate_mask(mask):
        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_g = Q.view(K, n_groups, group_size)
        W_deq = (Q_g - z) * s
        W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        Y_approx = X @ W_deq.T
        return ((Y_ref - Y_approx) ** 2).mean().item()

    mse_uniform = evaluate_mask(mask_uniform)
    row_sparsity_uniform = 1 - mask_uniform.mean(dim=1)
    print(f"\nUniform MSE: {mse_uniform:.6f}")
    print(f"Row sparsity range: [{row_sparsity_uniform.min():.2f}, {row_sparsity_uniform.max():.2f}]")

    # === SGSS: Variable sparsity based on μ₂ (row factors) ===
    print("\n--- SGSS: μ-based row sparsity ---")

    # μ₂ is [K, 1] - one per row (mu1 is [N] for columns)
    # High μ₂ = high variance row = can prune more
    mu2_flat = mu2.squeeze()  # [K]
    mu2_normalized = (mu2_flat - mu2_flat.min()) / (mu2_flat.max() - mu2_flat.min() + 1e-8)

    # Map μ to sparsity: high μ → high sparsity
    # sparsity_row = base + range * μ_normalized
    # We want average to be target_sparsity

    for sparsity_range in [0.1, 0.2, 0.3, 0.4]:
        min_sparsity = target_sparsity - sparsity_range / 2
        max_sparsity = target_sparsity + sparsity_range / 2

        # Per-row sparsity
        row_sparsity = min_sparsity + (max_sparsity - min_sparsity) * mu2_normalized
        row_sparsity = row_sparsity.clamp(0.1, 0.9)  # Safety bounds

        # Create mask: for each row, keep top (1 - row_sparsity[i]) weights
        mask_sgss = torch.zeros_like(W)
        for i in range(K):
            n_keep = max(1, int(N * (1 - row_sparsity[i].item())))
            top_idx = importance[i].topk(n_keep).indices
            mask_sgss[i, top_idx] = 1.0

        actual_sparsity = 1 - mask_sgss.mean().item()
        mse_sgss = evaluate_mask(mask_sgss)
        improvement = (mse_uniform - mse_sgss) / mse_uniform * 100

        print(f"Range ±{sparsity_range/2:.1%}: MSE={mse_sgss:.6f} ({improvement:+.2f}%) "
              f"[actual sparsity={actual_sparsity:.1%}]")

    # === Inverse SGSS: Low μ gets high sparsity ===
    print("\n--- Inverse SGSS: Low μ → high sparsity ---")
    for sparsity_range in [0.2, 0.4]:
        min_sparsity = target_sparsity - sparsity_range / 2
        max_sparsity = target_sparsity + sparsity_range / 2

        # Inverse: high μ = low sparsity
        row_sparsity = max_sparsity - (max_sparsity - min_sparsity) * mu2_normalized
        row_sparsity = row_sparsity.clamp(0.1, 0.9)

        mask_inv = torch.zeros_like(W)
        for i in range(K):
            n_keep = max(1, int(N * (1 - row_sparsity[i].item())))
            top_idx = importance[i].topk(n_keep).indices
            mask_inv[i, top_idx] = 1.0

        actual_sparsity = 1 - mask_inv.mean().item()
        mse_inv = evaluate_mask(mask_inv)
        improvement = (mse_uniform - mse_inv) / mse_uniform * 100

        print(f"Inv range ±{sparsity_range/2:.1%}: MSE={mse_inv:.6f} ({improvement:+.2f}%) "
              f"[actual sparsity={actual_sparsity:.1%}]")

    # === N:M structured sparsity comparison ===
    print("\n--- N:M Structured Sparsity (4:8 = 50%) ---")
    M = 8
    N_keep = 4
    mask_nm = torch.zeros_like(W)
    for i in range(K):
        for j in range(0, N, M):
            block_importance = importance[i, j:j+M]
            top_idx = block_importance.topk(N_keep).indices
            mask_nm[i, j + top_idx] = 1.0

    actual_sparsity_nm = 1 - mask_nm.mean().item()
    mse_nm = evaluate_mask(mask_nm)
    improvement_nm = (mse_uniform - mse_nm) / mse_uniform * 100
    print(f"4:8 N:M: MSE={mse_nm:.6f} ({improvement_nm:+.2f}%) [sparsity={actual_sparsity_nm:.1%}]")

    # === Variable N:M based on μ ===
    print("\n--- Variable N:M based on μ ---")
    M = 8
    mask_vnm = torch.zeros_like(W)

    # Map μ to N_keep: high μ → low N_keep (more pruning)
    # N_keep ranges from 2 to 6 (sparsity 25% to 75%)
    n_keep_per_row = 5 - (3 * mu2_normalized).round().int()  # Range [2, 5]
    n_keep_per_row = n_keep_per_row.clamp(2, 6)

    for i in range(K):
        n_keep_i = n_keep_per_row[i].item()
        for j in range(0, N, M):
            block_importance = importance[i, j:j+M]
            top_idx = block_importance.topk(n_keep_i).indices
            mask_vnm[i, j + top_idx] = 1.0

    actual_sparsity_vnm = 1 - mask_vnm.mean().item()
    mse_vnm = evaluate_mask(mask_vnm)
    improvement_vnm = (mse_uniform - mse_vnm) / mse_uniform * 100
    print(f"Variable N:M: MSE={mse_vnm:.6f} ({improvement_vnm:+.2f}%) [sparsity={actual_sparsity_vnm:.1%}]")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_sgss()
