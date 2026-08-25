"""
Hypothesis 38: Group-Aware Sparsity (GAS)

Core insight: Quantization computes scale per group (64 weights).
If a group has too many zeros from pruning, scale estimation is unreliable.

Standard approach: Global threshold → some groups may be mostly pruned
GAS: Ensure each group keeps at least k weights for robust scale estimation

Mathematical formulation:
- For each group, keep at least min_keep weights (e.g., 50% of group)
- Redistribute pruning budget to other groups
- This ensures all groups have enough data for scale estimation

Why this might help:
- Groups with few non-zeros have noisy scale estimates
- Scale error propagates to all weights in the group
- Ensuring minimum coverage reduces this variance
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_gas():
    print("="*70)
    print("GROUP-AWARE SPARSITY (GAS) TEST")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    test_layers = [
        ('gate_proj', model.model.layers[0].mlp.gate_proj.weight.data),
        ('q_proj', model.model.layers[0].self_attn.q_proj.weight.data),
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

        # Sinkhorn
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        act_norms = torch.norm(X, dim=0)
        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

        # === Standard: Global threshold ===
        threshold = importance.view(-1).sort().values[n_prune]
        mask_std = (importance > threshold).float()

        # Analyze per-group sparsity
        mask_grouped = mask_std.view(K, n_groups, group_size)
        per_group_kept = mask_grouped.sum(dim=2)  # [K, n_groups]
        print(f"\nStandard mask - per-group kept weights:")
        print(f"  Mean: {per_group_kept.mean():.1f}/{group_size}")
        print(f"  Min:  {per_group_kept.min():.0f}/{group_size}")
        print(f"  Max:  {per_group_kept.max():.0f}/{group_size}")
        print(f"  Groups with <50% kept: {(per_group_kept < group_size/2).sum().item()}")

        # === GAS: Ensure minimum per group ===
        def create_gas_mask(importance, min_keep_fraction):
            """Create mask ensuring each group keeps at least min_keep_fraction weights."""
            K, N = importance.shape
            n_groups = N // group_size
            min_keep = int(group_size * min_keep_fraction)

            # Reshape to groups
            imp_grouped = importance.view(K, n_groups, group_size)

            # For each group, identify which weights MUST be kept (top min_keep)
            mask = torch.zeros_like(importance)

            # First pass: ensure minimum per group
            for g in range(n_groups):
                group_imp = imp_grouped[:, g, :]  # [K, group_size]
                # For each row, keep top min_keep in this group
                for k in range(K):
                    _, top_idx = group_imp[k].topk(min_keep)
                    mask[k, g*group_size + top_idx] = 1.0

            # Count how many are "forced" to keep
            forced_keep = mask.sum().item()
            target_keep = K * N * (1 - sparsity)

            if forced_keep >= target_keep:
                # Already at or above target, just return
                return mask

            # Second pass: fill remaining budget with highest importance
            remaining_budget = int(target_keep - forced_keep)
            remaining_importance = importance * (1 - mask)  # Only consider not-yet-kept
            remaining_flat = remaining_importance.view(-1)
            if remaining_budget > 0:
                _, top_indices = remaining_flat.topk(remaining_budget)
                mask.view(-1)[top_indices] = 1.0

            return mask

        def evaluate_mask(mask, label):
            W_sparse = W_norm * mask
            Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
            Q_g = Q.view(K, n_groups, group_size)
            W_deq = (Q_g - z) * s
            W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y_approx = X @ W_deq.T
            mse = ((Y_ref - Y_approx) ** 2).mean().item()

            overlap = (mask * mask_std).sum() / mask_std.sum()
            actual_sparsity = 1 - mask.mean().item()

            return mse, overlap.item(), actual_sparsity

        mse_std, _, _ = evaluate_mask(mask_std, "Standard")
        print(f"\nStandard MSE: {mse_std:.6f}")

        # Test different minimum keep fractions
        for min_keep_frac in [0.3, 0.4, 0.5, 0.6]:
            mask_gas = create_gas_mask(importance, min_keep_frac)
            mse_gas, overlap, actual_sparsity = evaluate_mask(mask_gas, f"GAS-{min_keep_frac}")
            improvement = (mse_std - mse_gas) / mse_std * 100
            print(f"GAS min={min_keep_frac:.0%}: MSE={mse_gas:.6f} ({improvement:+.2f}%), "
                  f"overlap={overlap:.1%}, actual_sparsity={actual_sparsity:.1%}")

            # Check per-group distribution
            mask_grouped = mask_gas.view(K, n_groups, group_size)
            per_group_kept = mask_grouped.sum(dim=2)
            min_kept = per_group_kept.min().item()
            print(f"           Min kept per group: {min_kept:.0f}/{group_size}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_gas()
