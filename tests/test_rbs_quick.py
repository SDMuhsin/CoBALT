"""
Quick test: Row-Balanced Sparsity (RBS)

Observation: Output MSE depends on cumulative error per row.
High-norm rows accumulate more error when pruned.

Idea: Allocate different sparsity to different rows:
- High-norm rows (more sensitive): lower sparsity (keep more)
- Low-norm rows (less sensitive): higher sparsity (prune more)

This is different from per-weight importance - it's about
distributing sparsity across rows to minimize cumulative error.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_rbs():
    print("="*70)
    print("ROW-BALANCED SPARSITY (RBS) QUICK TEST")
    print("="*70)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load real weights
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    W = model.model.layers[0].mlp.gate_proj.weight.data.float()
    K, N = W.shape
    print(f"Weight shape: [{K}x{N}] (K=output rows, N=input cols)")

    # Create synthetic activations
    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1

    # Compute Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Compute norms
    act_norms = torch.norm(X, dim=0)  # [N]
    W_row_norms = torch.norm(W, dim=1)  # [K]

    # Target sparsity
    total_sparsity = 0.35
    total_weights = K * N
    target_pruned = int(total_weights * total_sparsity)

    # Method 1: Standard (global threshold)
    importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)
    threshold_std = importance_std.view(-1).sort().values[target_pruned]
    mask_std = (importance_std > threshold_std).float()

    # Method 2: Row-Balanced Sparsity
    # Allocate sparsity inversely proportional to row norm
    # High-norm rows: fewer pruned, Low-norm rows: more pruned
    row_weights = 1 / (W_row_norms + 1e-6)  # Inverse norm
    row_weights = row_weights / row_weights.sum() * K  # Normalize to sum to K

    # Sparsity per row: proportional to row weight (inverse norm)
    sparsity_per_row = total_sparsity * row_weights
    sparsity_per_row = sparsity_per_row.clamp(0.1, 0.7)  # Clip to reasonable range

    # Adjust to match total budget
    total_from_clip = (sparsity_per_row * N).sum()
    sparsity_per_row = sparsity_per_row * (target_pruned / total_from_clip)
    sparsity_per_row = sparsity_per_row.clamp(0.1, 0.7)

    # Create mask using per-row sparsity
    mask_rbs = torch.zeros_like(W)
    for i in range(K):
        row_importance = importance_std[i, :]  # Use same importance, different threshold
        n_prune_row = int(N * sparsity_per_row[i].item())
        n_prune_row = max(1, min(n_prune_row, N - 1))
        threshold = row_importance.sort().values[n_prune_row]
        mask_rbs[i, :] = (row_importance > threshold).float()

    # Method 3: Row-Balanced with μ₂ (row Sinkhorn factor)
    # High μ₂ means row needed more scaling = less important?
    row_weights_mu = 1 / (mu2 + 1e-6)
    row_weights_mu = row_weights_mu / row_weights_mu.sum() * K

    sparsity_per_row_mu = total_sparsity * row_weights_mu
    sparsity_per_row_mu = sparsity_per_row_mu.clamp(0.1, 0.7)
    total_from_clip_mu = (sparsity_per_row_mu * N).sum()
    sparsity_per_row_mu = sparsity_per_row_mu * (target_pruned / total_from_clip_mu)
    sparsity_per_row_mu = sparsity_per_row_mu.clamp(0.1, 0.7)

    mask_rbs_mu = torch.zeros_like(W)
    for i in range(K):
        row_importance = importance_std[i, :]
        n_prune_row = int(N * sparsity_per_row_mu[i].item())
        n_prune_row = max(1, min(n_prune_row, N - 1))
        threshold = row_importance.sort().values[n_prune_row]
        mask_rbs_mu[i, :] = (row_importance > threshold).float()

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

        # Weight MSE
        W_masked = W * mask
        w_mse = ((W_masked - W_deq * mask) ** 2).sum() / mask.sum()

        # Output MSE
        Y_approx = X @ W_deq.T
        out_mse = ((Y_ref - Y_approx) ** 2).mean()

        # Actual sparsity
        actual_sparsity = 1 - mask.mean().item()

        return w_mse.item(), out_mse.item(), actual_sparsity

    w_std, out_std, sp_std = evaluate_mask(mask_std, "Standard")
    w_rbs, out_rbs, sp_rbs = evaluate_mask(mask_rbs, "RBS-norm")
    w_rbs_mu, out_rbs_mu, sp_rbs_mu = evaluate_mask(mask_rbs_mu, "RBS-μ₂")

    print(f"\n--- Results (Target sparsity: {total_sparsity:.0%}) ---")
    print(f"{'Method':<15} {'Sparsity':<10} {'Weight MSE':<15} {'Output MSE':<15}")
    print(f"{'Standard':<15} {sp_std:<10.2%} {w_std:<15.6f} {out_std:<15.6f}")
    print(f"{'RBS-norm':<15} {sp_rbs:<10.2%} {w_rbs:<15.6f} ({(w_std-w_rbs)/w_std*100:+.2f}%) {out_rbs:<15.6f} ({(out_std-out_rbs)/out_std*100:+.2f}%)")
    print(f"{'RBS-μ₂':<15} {sp_rbs_mu:<10.2%} {w_rbs_mu:<15.6f} ({(w_std-w_rbs_mu)/w_std*100:+.2f}%) {out_rbs_mu:<15.6f} ({(out_std-out_rbs_mu)/out_std*100:+.2f}%)")

    # Analyze sparsity distribution
    actual_sp_std = 1 - mask_std.mean(dim=1)
    actual_sp_rbs = 1 - mask_rbs.mean(dim=1)
    actual_sp_rbs_mu = 1 - mask_rbs_mu.mean(dim=1)

    print(f"\n--- Sparsity Distribution (per row) ---")
    print(f"Standard:  mean={actual_sp_std.mean():.2%}, std={actual_sp_std.std():.2%}")
    print(f"RBS-norm:  mean={actual_sp_rbs.mean():.2%}, std={actual_sp_rbs.std():.2%}")
    print(f"RBS-μ₂:    mean={actual_sp_rbs_mu.mean():.2%}, std={actual_sp_rbs_mu.std():.2%}")

    # Correlation between row norm and sparsity
    corr_std = torch.corrcoef(torch.stack([W_row_norms, actual_sp_std]))[0, 1]
    corr_rbs = torch.corrcoef(torch.stack([W_row_norms, actual_sp_rbs]))[0, 1]
    corr_rbs_mu = torch.corrcoef(torch.stack([W_row_norms, actual_sp_rbs_mu]))[0, 1]

    print(f"\n--- Row norm vs sparsity correlation ---")
    print(f"Standard:  {corr_std:.4f} (ideally negative)")
    print(f"RBS-norm:  {corr_rbs:.4f}")
    print(f"RBS-μ₂:    {corr_rbs_mu:.4f}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_rbs()
