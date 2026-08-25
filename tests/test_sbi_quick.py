"""
Quick test: Sinkhorn-Balanced Importance (SBI)

NEW IDEA: Apply Sinkhorn to the IMPORTANCE matrix, not just weights.

Current approach:
1. Sinkhorn on W → μ factors
2. Compute importance = |W| × ||X|| / (μ₁ × μ₂)
3. Global threshold → mask

SBI approach:
1. Sinkhorn on W → μ factors
2. Compute raw_importance = |W| × ||X|| / (μ₁ × μ₂)
3. Sinkhorn on raw_importance → balanced_importance
4. Global threshold on balanced_importance → mask

The key insight: Sinkhorn on importance ensures that importance is
distributed evenly across rows and columns, leading to more uniform
sparsity pattern. This prevents "all-or-nothing" rows.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_sbi():
    print("="*70)
    print("SINKHORN-BALANCED IMPORTANCE (SBI) QUICK TEST")
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
    print(f"Weight shape: [{K}x{N}]")

    # Create activations
    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1

    # Compute Sinkhorn on weights
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Compute activation norms
    act_norms = torch.norm(X, dim=0)

    # Standard importance (inverse μ)
    importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # SBI: Apply Sinkhorn to importance
    # Need to ensure importance is positive (it is since |W| > 0)
    importance_clipped = importance_std.clamp(min=1e-8)
    importance_balanced, mu1_imp, mu2_imp = sinkhorn_log(importance_clipped, order=16)
    mu2_imp = mu2_imp.squeeze()

    # The balanced importance
    importance_sbi = importance_balanced

    # Also try: use original importance but with SBI-derived μ to weight differently
    importance_sbi_v2 = importance_std / (mu1_imp.unsqueeze(0) * mu2_imp.unsqueeze(1) + 1e-6)

    # Create masks (35% sparsity)
    sparsity = 0.35
    n_prune = int(K * N * sparsity)

    def create_mask(importance):
        threshold = importance.view(-1).sort().values[n_prune]
        return (importance > threshold).float()

    mask_std = create_mask(importance_std)
    mask_sbi = create_mask(importance_sbi)
    mask_sbi_v2 = create_mask(importance_sbi_v2)

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

        # Sparsity distribution
        sp_per_row = 1 - mask.mean(dim=1)

        return w_mse.item(), out_mse.item(), sp_per_row

    w_std, out_std, sp_std = evaluate_mask(mask_std, "Standard")
    w_sbi, out_sbi, sp_sbi = evaluate_mask(mask_sbi, "SBI")
    w_sbi2, out_sbi2, sp_sbi2 = evaluate_mask(mask_sbi_v2, "SBI-v2")

    print(f"\n--- Results ---")
    print(f"{'Method':<15} {'Weight MSE':<15} {'Output MSE':<20}")
    print(f"{'Standard':<15} {w_std:<15.6f} {out_std:<20.6f}")
    print(f"{'SBI':<15} {w_sbi:<15.6f} ({(w_std-w_sbi)/w_std*100:+.2f}%) {out_sbi:<20.6f} ({(out_std-out_sbi)/out_std*100:+.2f}%)")
    print(f"{'SBI-v2':<15} {w_sbi2:<15.6f} ({(w_std-w_sbi2)/w_std*100:+.2f}%) {out_sbi2:<20.6f} ({(out_std-out_sbi2)/out_std*100:+.2f}%)")

    # Analyze sparsity distribution
    print(f"\n--- Sparsity Distribution (per row) ---")
    print(f"Standard:  mean={sp_std.mean():.2%}, std={sp_std.std():.2%}")
    print(f"SBI:       mean={sp_sbi.mean():.2%}, std={sp_sbi.std():.2%}")
    print(f"SBI-v2:    mean={sp_sbi2.mean():.2%}, std={sp_sbi2.std():.2%}")

    # Analyze importance distribution before/after balancing
    print(f"\n--- Importance Distribution ---")
    print(f"Before SBI: row CV={importance_std.std(dim=1).mean()/importance_std.mean():.4f}, col CV={importance_std.std(dim=0).mean()/importance_std.mean():.4f}")
    print(f"After SBI:  row CV={importance_sbi.std(dim=1).mean()/importance_sbi.mean():.4f}, col CV={importance_sbi.std(dim=0).mean()/importance_sbi.mean():.4f}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_sbi()
