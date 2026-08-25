"""
Quick test: Balanced Sparsity (BS)

KEY INSIGHT: Sinkhorn creates balanced row/column variance.
Unstructured pruning BREAKS this balance - some rows lose more weights than others.

Balanced Sparsity: Enforce that each row AND each column has exactly
the target sparsity. This preserves Sinkhorn's balance property!

This is different from:
- Unstructured: Global ranking, uneven per-row/col sparsity
- Structured (N:M): Fixed pattern, removes flexibility
- Balanced: Unstructured positions, but equal per-row/col count

Algorithm:
1. For each row, keep top (1-sparsity) fraction by importance
2. For each column, keep top (1-sparsity) fraction by importance
3. Final mask = row_mask AND col_mask (must survive both)
4. This gives ~sparsity² effective sparsity, so adjust

Alternative: Use bipartite matching to find optimal mask with constraints.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_balanced_sparsity():
    print("="*70)
    print("BALANCED SPARSITY (BS) TEST")
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

    # Activations
    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
    Y_ref = X @ W.T

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Importance (inverse μ)
    act_norms = torch.norm(X, dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # === Standard unstructured ===
    n_prune = int(K * N * sparsity)
    threshold_std = importance.view(-1).sort().values[n_prune]
    mask_std = (importance > threshold_std).float()

    # Check balance
    row_sparsity_std = 1 - mask_std.mean(dim=1)
    col_sparsity_std = 1 - mask_std.mean(dim=0)
    print(f"\n--- Standard Unstructured ---")
    print(f"Row sparsity: mean={row_sparsity_std.mean():.3f}, std={row_sparsity_std.std():.3f}")
    print(f"Col sparsity: mean={col_sparsity_std.mean():.3f}, std={col_sparsity_std.std():.3f}")

    # === Row-balanced: each row has exactly target sparsity ===
    n_keep_per_row = int(N * (1 - sparsity))
    mask_row = torch.zeros_like(W)
    for i in range(K):
        row_importance = importance[i]
        top_k = row_importance.topk(n_keep_per_row).indices
        mask_row[i, top_k] = 1.0

    row_sparsity_rb = 1 - mask_row.mean(dim=1)
    col_sparsity_rb = 1 - mask_row.mean(dim=0)
    print(f"\n--- Row-Balanced ---")
    print(f"Row sparsity: mean={row_sparsity_rb.mean():.3f}, std={row_sparsity_rb.std():.3f}")
    print(f"Col sparsity: mean={col_sparsity_rb.mean():.3f}, std={col_sparsity_rb.std():.3f}")

    # === Col-balanced: each column has exactly target sparsity ===
    n_keep_per_col = int(K * (1 - sparsity))
    mask_col = torch.zeros_like(W)
    for j in range(N):
        col_importance = importance[:, j]
        top_k = col_importance.topk(n_keep_per_col).indices
        mask_col[top_k, j] = 1.0

    row_sparsity_cb = 1 - mask_col.mean(dim=1)
    col_sparsity_cb = 1 - mask_col.mean(dim=0)
    print(f"\n--- Col-Balanced ---")
    print(f"Row sparsity: mean={row_sparsity_cb.mean():.3f}, std={row_sparsity_cb.std():.3f}")
    print(f"Col sparsity: mean={col_sparsity_cb.mean():.3f}, std={col_sparsity_cb.std():.3f}")

    # === Doubly-balanced: Intersection of row and col balanced ===
    mask_both = mask_row * mask_col
    actual_sparsity_both = 1 - mask_both.mean().item()
    print(f"\n--- Doubly-Balanced (intersection) ---")
    print(f"Actual sparsity: {actual_sparsity_both:.1%} (target was {sparsity:.1%})")

    # === Alternating projection: Iteratively balance rows and cols ===
    print(f"\n--- Alternating Projection (iterative) ---")
    mask_alt = importance.clone()
    target_keep = 1 - sparsity

    for iteration in range(10):
        # Row normalization: scale each row to have correct mean
        row_means = mask_alt.mean(dim=1, keepdim=True)
        mask_alt = mask_alt * (target_keep / (row_means + 1e-8))

        # Col normalization: scale each col to have correct mean
        col_means = mask_alt.mean(dim=0, keepdim=True)
        mask_alt = mask_alt * (target_keep / (col_means + 1e-8))

        # Clamp to [0, 1]
        mask_alt = mask_alt.clamp(0, 1)

    # Threshold to get binary mask
    threshold_alt = mask_alt.view(-1).sort().values[n_prune]
    mask_alt_binary = (mask_alt > threshold_alt).float()

    row_sparsity_alt = 1 - mask_alt_binary.mean(dim=1)
    col_sparsity_alt = 1 - mask_alt_binary.mean(dim=0)
    print(f"Row sparsity: mean={row_sparsity_alt.mean():.3f}, std={row_sparsity_alt.std():.3f}")
    print(f"Col sparsity: mean={col_sparsity_alt.mean():.3f}, std={col_sparsity_alt.std():.3f}")

    # === Evaluate all masks ===
    def evaluate_mask(mask, name):
        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_g = Q.view(K, n_groups, group_size)
        W_deq = (Q_g - z) * s
        W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        Y_approx = X @ W_deq.T
        return ((Y_ref - Y_approx) ** 2).mean().item()

    mse_std = evaluate_mask(mask_std, "Standard")
    mse_row = evaluate_mask(mask_row, "Row-balanced")
    mse_col = evaluate_mask(mask_col, "Col-balanced")
    mse_both = evaluate_mask(mask_both, "Doubly-balanced")
    mse_alt = evaluate_mask(mask_alt_binary, "Alternating")

    print(f"\n{'='*70}")
    print("MSE RESULTS")
    print("="*70)
    print(f"Standard:        {mse_std:.6f}")
    print(f"Row-balanced:    {mse_row:.6f} ({(mse_std-mse_row)/mse_std*100:+.2f}%)")
    print(f"Col-balanced:    {mse_col:.6f} ({(mse_std-mse_col)/mse_std*100:+.2f}%)")
    print(f"Doubly-balanced: {mse_both:.6f} ({(mse_std-mse_both)/mse_std*100:+.2f}%) [sparsity={1-mask_both.mean():.1%}]")
    print(f"Alternating:     {mse_alt:.6f} ({(mse_std-mse_alt)/mse_std*100:+.2f}%)")

    # === Sinkhorn-based balanced sparsity ===
    print(f"\n{'='*70}")
    print("SINKHORN-BASED BALANCED MASK")
    print("="*70)

    # Use Sinkhorn to balance the importance matrix!
    # This finds weights that are important from BOTH row and col perspective
    importance_norm, imp_mu1, imp_mu2 = sinkhorn_log(importance, order=32)
    imp_mu2 = imp_mu2.squeeze()

    # Global threshold on balanced importance
    threshold_sink = importance_norm.view(-1).sort().values[n_prune]
    mask_sink = (importance_norm > threshold_sink).float()

    row_sparsity_sink = 1 - mask_sink.mean(dim=1)
    col_sparsity_sink = 1 - mask_sink.mean(dim=0)
    print(f"Row sparsity: mean={row_sparsity_sink.mean():.3f}, std={row_sparsity_sink.std():.3f}")
    print(f"Col sparsity: mean={col_sparsity_sink.mean():.3f}, std={col_sparsity_sink.std():.3f}")

    mse_sink = evaluate_mask(mask_sink, "Sinkhorn-balanced")
    print(f"Sinkhorn-balanced MSE: {mse_sink:.6f} ({(mse_std-mse_sink)/mse_std*100:+.2f}%)")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_balanced_sparsity()
