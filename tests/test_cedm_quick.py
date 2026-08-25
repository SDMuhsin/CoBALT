"""
Hypothesis 39: Calibration-Error-Driven Mask (CEDM)

Core insight: Importance heuristics (|W| × ||X||) are PROXIES for output error.
Why not directly measure output error?

Approach:
1. Use calibration data to compute Y_ref = X @ W
2. For each weight, compute the output error if it's pruned
3. Prune weights with lowest error contribution

This is computationally expensive but theoretically optimal.
We can approximate by:
- Computing per-weight sensitivity: ΔY / ΔW
- This is essentially the output gradient

Mathematical formulation:
- Output contribution of w[i,j]: Y[:, i] ≈ X[:, j] × W[i, j]
- Pruning w[i,j] changes output by: ΔY[:, i] = -X[:, j] × W[i, j]
- Error from pruning: ||ΔY||² = ||X[:, j]||² × |W[i,j]|²

Wait - this is exactly |W| × ||X||! The standard importance IS the output error.

Let me try something different: Account for QUANTIZATION in the error.
The true error from pruning + quantizing is:
||X @ Q(W_sparse) - X @ W||

This depends on how quantization changes with the mask.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_cedm():
    print("="*70)
    print("CALIBRATION-ERROR-DRIVEN MASK (CEDM) TEST")
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

    batch = 128  # Larger batch for better calibration
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
    Y_ref = X @ W.T

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    act_norms = torch.norm(X, dim=0)
    importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # Standard mask
    threshold_std = importance_std.view(-1).sort().values[n_prune]
    mask_std = (importance_std > threshold_std).float()

    def compute_quantized_error(mask):
        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_g = Q.view(K, n_groups, group_size)
        W_deq = (Q_g - z) * s
        W_deq = W_deq.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)
        Y_approx = X @ W_deq.T
        return ((Y_ref - Y_approx) ** 2).mean().item()

    mse_std = compute_quantized_error(mask_std)
    print(f"Standard MSE: {mse_std:.6f}")

    # === CEDM: Greedy pruning based on actual quantized error ===
    print("\n--- CEDM: Greedy Error-Based Pruning ---")

    # Start with all weights kept
    mask_cedm = torch.ones_like(W)

    # Greedy: Remove weights one at a time (or in batches)
    # This is slow, so we'll do it in batches

    batch_prune = 1000  # Prune 1000 at a time
    n_batches = n_prune // batch_prune

    # Compute initial quantized error for each potential prune
    # This is approximated by the marginal contribution

    for batch_idx in range(min(5, n_batches)):  # Only do a few batches for speed
        current_kept = (mask_cedm > 0)
        n_current = current_kept.sum().item()

        # For each kept weight, estimate error if pruned
        # Approximation: use importance + quantization effect

        # Simulate: for each weight, what's the error increase if removed?
        # This is expensive, so we sample
        n_samples = min(500, int(n_current))
        kept_indices = current_kept.nonzero()
        sample_indices = kept_indices[torch.randperm(len(kept_indices))[:n_samples]]

        prune_errors = torch.zeros(n_samples, device=W.device)

        current_mse = compute_quantized_error(mask_cedm)

        for i, (ki, kj) in enumerate(sample_indices):
            # Temporarily prune this weight
            mask_temp = mask_cedm.clone()
            mask_temp[ki, kj] = 0
            temp_mse = compute_quantized_error(mask_temp)
            prune_errors[i] = temp_mse - current_mse

        # Find weights with lowest error increase
        _, best_prune = prune_errors.topk(min(batch_prune, n_samples), largest=False)

        # Prune those weights
        for idx in best_prune:
            ki, kj = sample_indices[idx]
            mask_cedm[ki, kj] = 0

        new_mse = compute_quantized_error(mask_cedm)
        actual_pruned = (mask_cedm == 0).sum().item()
        print(f"  Batch {batch_idx}: pruned {actual_pruned} weights, MSE = {new_mse:.6f}")

    # Fill remaining with importance-based pruning
    remaining_to_prune = n_prune - (mask_cedm == 0).sum().item()
    if remaining_to_prune > 0:
        remaining_importance = importance_std * mask_cedm
        remaining_flat = remaining_importance.view(-1)
        # Get the remaining_to_prune lowest importance among kept
        kept_importance = remaining_flat[remaining_flat > 0]
        if len(kept_importance) > remaining_to_prune:
            thresh = kept_importance.sort().values[remaining_to_prune]
            prune_mask = (remaining_importance > 0) & (remaining_importance <= thresh)
            mask_cedm[prune_mask] = 0

    mse_cedm = compute_quantized_error(mask_cedm)
    overlap = (mask_cedm * mask_std).sum() / mask_std.sum()
    actual_sparsity = (mask_cedm == 0).sum().item() / (K * N)

    print(f"\nFinal CEDM MSE: {mse_cedm:.6f} ({(mse_std-mse_cedm)/mse_std*100:+.2f}%)")
    print(f"Overlap with standard: {overlap:.1%}")
    print(f"Actual sparsity: {actual_sparsity:.1%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_cedm()
