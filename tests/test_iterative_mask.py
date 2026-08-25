"""
Quick test: Iterative Mask Refinement (IMR)

KEY INSIGHT: Initial importance-based mask is computed WITHOUT knowing
the actual quantization error. What if we refine the mask iteratively?

Algorithm:
1. Create initial mask from importance
2. Quantize → measure per-weight contribution to error
3. Identify "bad" kept weights (high error contribution)
4. Identify "good" pruned weights (would reduce error if restored)
5. Swap: prune bad, restore good
6. Repeat until convergence

This is like coordinate descent on the mask.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_imr():
    print("="*70)
    print("ITERATIVE MASK REFINEMENT (IMR) TEST")
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

    # Activations
    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
    Y_ref = X @ W.T

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Initial importance and mask
    act_norms = torch.norm(X, dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)
    threshold = importance.view(-1).sort().values[n_prune]
    mask = (importance > threshold).float()

    def compute_mse(mask):
        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_g = Q.view(K, n_groups, group_size)
        W_deq = (Q_g - z) * s
        W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        Y_approx = X @ W_deq.T
        return ((Y_ref - Y_approx) ** 2).mean().item(), W_deq

    mse_initial, _ = compute_mse(mask)
    print(f"\nInitial MSE: {mse_initial:.6f}")

    # Iterative refinement
    best_mask = mask.clone()
    best_mse = mse_initial

    print("\n--- Iterative Refinement ---")

    for iteration in range(5):
        # Compute current quantized weights
        current_mse, W_deq_current = compute_mse(best_mask)

        # Per-weight error contribution (approximation)
        # Error from weight (i,j) ≈ (W_deq[i,j] - W_true[i,j]) * X[:, j]
        W_true = W_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        weight_error = (W_deq_current - W_true).abs()

        # Scale by activation magnitude
        act_scale = X.abs().mean(dim=0)  # [N]
        weighted_error = weight_error * act_scale.unsqueeze(0)  # [K, N]

        # Find worst kept weights (high error, currently kept)
        kept_mask = best_mask > 0
        error_kept = weighted_error.clone()
        error_kept[~kept_mask] = -float('inf')

        # Find best pruned weights (low importance but worth restoring)
        pruned_mask = best_mask == 0
        # For pruned weights, estimate error if restored
        # Approximation: use original importance as proxy
        restore_benefit = importance.clone()
        restore_benefit[~pruned_mask] = float('inf')

        # Try swapping worst kept with best pruned
        n_swaps = min(100, n_prune // 10)  # Swap up to 10% of pruned weights

        # Find worst kept
        _, worst_kept_idx = error_kept.view(-1).topk(n_swaps)
        # Find best pruned (lowest restore_benefit = easiest to restore)
        _, best_pruned_idx = restore_benefit.view(-1).topk(n_swaps, largest=False)

        # Try the swap
        new_mask = best_mask.clone()
        new_mask.view(-1)[worst_kept_idx] = 0  # Prune the worst
        new_mask.view(-1)[best_pruned_idx] = 1  # Restore the best

        new_mse, _ = compute_mse(new_mask)

        if new_mse < best_mse:
            improvement = (best_mse - new_mse) / best_mse * 100
            print(f"Iter {iteration+1}: MSE {new_mse:.6f} ({improvement:+.2f}%) - {n_swaps} swaps accepted")
            best_mask = new_mask
            best_mse = new_mse
        else:
            print(f"Iter {iteration+1}: No improvement, reducing swap count")
            n_swaps = n_swaps // 2
            if n_swaps < 10:
                break

    print(f"\n--- Summary ---")
    print(f"Initial MSE: {mse_initial:.6f}")
    print(f"Final MSE:   {best_mse:.6f}")
    improvement = (mse_initial - best_mse) / mse_initial * 100
    print(f"Improvement: {improvement:+.2f}%")

    # Also try: random swaps baseline
    print("\n--- Random Swap Baseline ---")
    random_mask = mask.clone()
    random_mse = mse_initial

    for iteration in range(5):
        n_swaps = 100
        new_mask = random_mask.clone()

        # Random swaps
        kept_idx = (random_mask > 0).nonzero(as_tuple=True)
        pruned_idx = (random_mask == 0).nonzero(as_tuple=True)

        perm_kept = torch.randperm(len(kept_idx[0]))[:n_swaps]
        perm_pruned = torch.randperm(len(pruned_idx[0]))[:n_swaps]

        for i in range(min(n_swaps, len(perm_kept), len(perm_pruned))):
            ki, kj = kept_idx[0][perm_kept[i]], kept_idx[1][perm_kept[i]]
            pi, pj = pruned_idx[0][perm_pruned[i]], pruned_idx[1][perm_pruned[i]]
            new_mask[ki, kj] = 0
            new_mask[pi, pj] = 1

        new_mse, _ = compute_mse(new_mask)
        if new_mse < random_mse:
            print(f"Random iter {iteration+1}: MSE {new_mse:.6f}")
            random_mask = new_mask
            random_mse = new_mse
        else:
            print(f"Random iter {iteration+1}: No improvement")

    print(f"\nRandom baseline final: {random_mse:.6f}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_imr()
