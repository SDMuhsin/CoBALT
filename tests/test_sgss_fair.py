"""
Fair test of SGSS: Enforce EXACT target sparsity

The previous test showed improvement but at lower sparsity.
This test ensures we compare at exactly the same sparsity level.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_sgss_fair():
    print("="*70)
    print("FAIR SGSS TEST: EQUAL SPARSITY COMPARISON")
    print("="*70)

    torch.manual_seed(42)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16, device_map="auto"
    )

    test_layers = [
        ('gate_proj', model.model.layers[0].mlp.gate_proj.weight.data),
        ('q_proj', model.model.layers[0].self_attn.q_proj.weight.data),
        ('k_proj', model.model.layers[0].self_attn.k_proj.weight.data),
    ]

    target_sparsity = 0.35
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
        n_prune = int(K * N * target_sparsity)

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
        row_sp = 1 - mask_uniform.mean(dim=1)
        print(f"Uniform: MSE={mse_uniform:.6f}, row sparsity std={row_sp.std():.4f}")

        # === SGSS: Variable per-row sparsity, same TOTAL ===
        # μ₂ normalized for row sparsity
        mu2_flat = mu2.squeeze()
        mu2_norm = (mu2_flat - mu2_flat.min()) / (mu2_flat.max() - mu2_flat.min() + 1e-8)

        # Strategy: Assign per-row QUOTA based on μ
        # High μ → more quota to prune (higher n_prune_per_row)
        # But total n_prune_per_row should equal n_prune

        for scale in [0.3, 0.5, 0.7, 1.0]:
            # Per-row weight for pruning budget: proportional to μ^scale
            prune_weight = mu2_norm ** scale  # Higher μ → higher weight → more pruning
            prune_weight = prune_weight / prune_weight.sum() * n_prune  # Normalize to total budget

            # Round to integers
            n_prune_per_row = prune_weight.round().int()
            # Adjust to hit exact target
            diff = n_prune - n_prune_per_row.sum().item()
            if diff > 0:
                # Need to prune more, add to high-μ rows
                indices = mu2_norm.argsort(descending=True)[:abs(diff)]
                n_prune_per_row[indices] += 1
            elif diff < 0:
                # Need to prune less, remove from low-μ rows
                indices = mu2_norm.argsort()[:abs(diff)]
                n_prune_per_row[indices] -= 1

            # Clamp to valid range
            n_prune_per_row = n_prune_per_row.clamp(0, N-1)

            # Create mask
            mask_sgss = torch.ones_like(W)
            for i in range(K):
                n_to_prune = n_prune_per_row[i].item()
                if n_to_prune > 0:
                    # Prune lowest importance in this row
                    _, prune_idx = importance[i].topk(n_to_prune, largest=False)
                    mask_sgss[i, prune_idx] = 0

            actual_sparsity = 1 - mask_sgss.mean().item()
            mse_sgss = evaluate_mask(mask_sgss)
            improvement = (mse_uniform - mse_sgss) / mse_uniform * 100

            row_sp_sgss = 1 - mask_sgss.mean(dim=1)
            overlap = (mask_uniform * mask_sgss).sum() / mask_uniform.sum()

            print(f"SGSS scale={scale}: MSE={mse_sgss:.6f} ({improvement:+.2f}%) "
                  f"[sparsity={actual_sparsity:.1%}, row_std={row_sp_sgss.std():.4f}, overlap={overlap:.1%}]")

        # === Alternative: Inverse SGSS (low μ gets more pruning) ===
        print("--- Inverse (low μ → more pruning) ---")
        for scale in [0.5, 1.0]:
            prune_weight = (1 - mu2_norm) ** scale
            prune_weight = prune_weight / prune_weight.sum() * n_prune

            n_prune_per_row = prune_weight.round().int()
            diff = n_prune - n_prune_per_row.sum().item()
            if diff > 0:
                indices = (1-mu2_norm).argsort(descending=True)[:abs(diff)]
                n_prune_per_row[indices] += 1
            elif diff < 0:
                indices = (1-mu2_norm).argsort()[:abs(diff)]
                n_prune_per_row[indices] -= 1

            n_prune_per_row = n_prune_per_row.clamp(0, N-1)

            mask_inv = torch.ones_like(W)
            for i in range(K):
                n_to_prune = n_prune_per_row[i].item()
                if n_to_prune > 0:
                    _, prune_idx = importance[i].topk(n_to_prune, largest=False)
                    mask_inv[i, prune_idx] = 0

            actual_sparsity = 1 - mask_inv.mean().item()
            mse_inv = evaluate_mask(mask_inv)
            improvement = (mse_uniform - mse_inv) / mse_uniform * 100

            print(f"Inv scale={scale}: MSE={mse_inv:.6f} ({improvement:+.2f}%) [sparsity={actual_sparsity:.1%}]")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_sgss_fair()
