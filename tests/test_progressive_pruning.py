"""
Quick test: Progressive Pruning with Sinkhorn Recalibration (PPSR)

KEY INSIGHT: One-shot pruning computes μ on full matrix, then prunes.
But pruning changes the variance structure, making μ suboptimal.

PPSR: Prune gradually, recalibrating Sinkhorn at each step.
1. Prune 5-10% of remaining weights
2. Recompute Sinkhorn on partially-sparse matrix
3. Use new μ for next pruning iteration
4. Repeat until target sparsity

Why this might help:
- μ adapts to sparse structure progressively
- Pruning decisions account for already-pruned weights
- More robust importance estimation
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_ppsr():
    print("="*70)
    print("PROGRESSIVE PRUNING + SINKHORN RECALIBRATION (PPSR) TEST")
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

        batch = 64
        X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
        Y_ref = X @ W.T
        act_norms = torch.norm(X, dim=0)

        # === One-shot pruning (baseline) ===
        W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
        mu2 = mu2.squeeze()

        importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)
        n_prune = int(K * N * target_sparsity)
        threshold = importance.view(-1).sort().values[n_prune]
        mask_oneshot = (importance > threshold).float()

        W_sparse = W_norm * mask_oneshot
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_g = Q.view(K, n_groups, group_size)
        W_deq = (Q_g - z) * s
        W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        Y_oneshot = X @ W_deq.T
        mse_oneshot = ((Y_ref - Y_oneshot) ** 2).mean().item()
        print(f"One-shot MSE: {mse_oneshot:.6f}")

        # === Progressive pruning ===
        for n_steps in [3, 5, 10]:
            mask_prog = torch.ones_like(W)
            W_current = W.clone()
            total_pruned = 0
            target_pruned = int(K * N * target_sparsity)

            for step in range(n_steps):
                # How many to prune this step
                remaining_to_prune = target_pruned - total_pruned
                current_remaining = mask_prog.sum().item()
                prune_this_step = min(
                    int(remaining_to_prune / (n_steps - step)),
                    int(current_remaining * 0.5)  # Don't prune more than 50% in one step
                )

                if prune_this_step <= 0:
                    break

                # Compute Sinkhorn on current (partially sparse) matrix
                W_masked = W * mask_prog
                # For Sinkhorn, set zeros to small values
                W_for_sink = W_masked.clone()
                W_for_sink[mask_prog == 0] = 1e-10

                W_norm_step, mu1_step, mu2_step = sinkhorn_log(W_for_sink, order=16)
                mu2_step = mu2_step.squeeze()

                # Importance on remaining weights
                importance_step = W.abs() * act_norms.unsqueeze(0) / (mu1_step.unsqueeze(0) * mu2_step.unsqueeze(1) + 1e-6)
                importance_step = importance_step * mask_prog  # Only consider remaining

                # Find lowest importance among remaining
                importance_flat = importance_step.view(-1)
                remaining_importance = importance_flat[importance_flat > 0]
                threshold_step = remaining_importance.sort().values[prune_this_step]

                # Prune
                new_prune = (importance_step > 0) & (importance_step <= threshold_step)
                mask_prog[new_prune] = 0
                total_pruned += new_prune.sum().item()

            actual_sparsity = 1 - mask_prog.mean().item()

            # Quantize with final Sinkhorn
            W_final = W * mask_prog
            W_for_sink = W_final.clone()
            W_for_sink[mask_prog == 0] = 1e-10
            W_norm_final, mu1_final, mu2_final = sinkhorn_log(W_for_sink, order=16)
            mu2_final = mu2_final.squeeze()

            W_sparse_prog = W_norm_final * mask_prog
            Q_prog, s_prog, z_prog, _ = quantize_rtn(W_sparse_prog, min_max, group_size=group_size)
            Q_g_prog = Q_prog.view(K, n_groups, group_size)
            W_deq_prog = (Q_g_prog - z_prog) * s_prog
            W_deq_prog = W_deq_prog.view(K, N) * mu2_final.unsqueeze(1) * mu1_final.unsqueeze(0)
            Y_prog = X @ W_deq_prog.T
            mse_prog = ((Y_ref - Y_prog) ** 2).mean().item()

            improvement = (mse_oneshot - mse_prog) / mse_oneshot * 100
            print(f"Progressive ({n_steps} steps, {actual_sparsity:.1%}): {mse_prog:.6f} ({improvement:+.2f}%)")

        # === Alternative: Use original μ but progressive mask ===
        print("--- Progressive with ORIGINAL μ (no recalibration) ---")
        for n_steps in [5, 10]:
            # Use original μ throughout
            mask_prog2 = torch.ones_like(W)
            total_pruned = 0
            target_pruned = int(K * N * target_sparsity)

            for step in range(n_steps):
                remaining_to_prune = target_pruned - total_pruned
                current_remaining = mask_prog2.sum().item()
                prune_this_step = min(
                    int(remaining_to_prune / (n_steps - step)),
                    int(current_remaining * 0.5)
                )

                if prune_this_step <= 0:
                    break

                # Use ORIGINAL importance but only on remaining
                importance_step = importance * mask_prog2

                importance_flat = importance_step.view(-1)
                remaining_importance = importance_flat[importance_flat > 0]
                threshold_step = remaining_importance.sort().values[prune_this_step]

                new_prune = (importance_step > 0) & (importance_step <= threshold_step)
                mask_prog2[new_prune] = 0
                total_pruned += new_prune.sum().item()

            actual_sparsity = 1 - mask_prog2.mean().item()

            W_sparse2 = W_norm * mask_prog2
            Q2, s2, z2, _ = quantize_rtn(W_sparse2, min_max, group_size=group_size)
            Q_g2 = Q2.view(K, n_groups, group_size)
            W_deq2 = (Q_g2 - z2) * s2
            W_deq2 = W_deq2.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
            Y2 = X @ W_deq2.T
            mse2 = ((Y_ref - Y2) ** 2).mean().item()

            # Check mask overlap with one-shot
            overlap = (mask_prog2 * mask_oneshot).sum() / mask_oneshot.sum()

            improvement = (mse_oneshot - mse2) / mse_oneshot * 100
            print(f"Prog-orig ({n_steps} steps): {mse2:.6f} ({improvement:+.2f}%) [overlap={overlap:.1%}]")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_ppsr()
