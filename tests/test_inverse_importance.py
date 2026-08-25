"""
Test: Inverse μ weighting for importance.

Current SINQ-Sparse: importance = |W| × ||X|| × μ₁ × μ₂
  → Keeps high-μ weights (which have amplified errors)

Alternative: importance = |W| × ||X|| / (μ₁ × μ₂)
  → Keeps low-μ weights (which have smaller error amplification)

This is the OPPOSITE of current approach. Does it help?
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_inverse_importance():
    print("="*70)
    print("INVERSE μ IMPORTANCE TEST")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print("\nLoading Qwen-0.5B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Get one layer's weight
    W = model.model.layers[0].mlp.gate_proj.weight.data.float()
    K, N = W.shape
    print(f"Weight shape: [{K}x{N}]")

    # Create synthetic activations
    torch.manual_seed(42)
    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
    act_norms = torch.norm(X, dim=0)

    # Compute Sinkhorn factors
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # Method 1: Standard sinq_wanda importance
    importance_std = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2.unsqueeze(1)

    # Method 2: Inverse μ importance
    importance_inv = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # Method 3: Pure Wanda (no μ)
    importance_wanda = W.abs() * act_norms.unsqueeze(0)

    # Create masks (35% sparsity)
    sparsity = 0.35
    n_prune = int(K * N * sparsity)

    def create_mask(importance):
        threshold = importance.view(-1).sort().values[n_prune]
        return (importance > threshold).float()

    mask_std = create_mask(importance_std)
    mask_inv = create_mask(importance_inv)
    mask_wanda = create_mask(importance_wanda)

    print(f"\nSparsity: {sparsity:.0%}")
    print(f"Weights pruned: {n_prune:,}")

    # Check mask overlap
    overlap_std_inv = (mask_std * mask_inv).sum() / mask_std.sum()
    overlap_std_wanda = (mask_std * mask_wanda).sum() / mask_std.sum()

    print(f"\nMask overlap with standard:")
    print(f"  Inverse μ: {overlap_std_inv:.1%}")
    print(f"  Pure Wanda: {overlap_std_wanda:.1%}")

    # Quantize and measure reconstruction MSE
    min_max = [0, 7]  # 3-bit
    group_size = 64

    def quantize_and_measure(W, W_norm, mask, mu1, mu2, min_max, group_size):
        """Quantize sparse matrix and measure reconstruction MSE."""
        W_sparse_norm = W_norm * mask

        Q, scales, zeros, _ = quantize_rtn(W_sparse_norm, min_max, group_size=group_size)

        # Dequantize
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)

        W_deq = W_deq_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0)

        # MSE on non-zero entries only
        W_masked = W * mask
        mse = ((W_masked - W_deq * mask) ** 2).sum() / mask.sum()

        return mse.item()

    mse_std = quantize_and_measure(W, W_norm, mask_std, mu1, mu2, min_max, group_size)
    mse_inv = quantize_and_measure(W, W_norm, mask_inv, mu1, mu2, min_max, group_size)
    mse_wanda = quantize_and_measure(W, W_norm, mask_wanda, mu1, mu2, min_max, group_size)

    print(f"\n--- Reconstruction MSE ---")
    print(f"Standard (|W|×||X||×μ₁×μ₂): {mse_std:.6f}")
    print(f"Inverse (|W|×||X||/μ₁/μ₂):  {mse_inv:.6f}")
    print(f"Pure Wanda (|W|×||X||):      {mse_wanda:.6f}")

    improvement_inv = (mse_std - mse_inv) / mse_std * 100
    improvement_wanda = (mse_std - mse_wanda) / mse_std * 100

    print(f"\nInverse vs Standard: {improvement_inv:+.2f}%")
    print(f"Wanda vs Standard: {improvement_wanda:+.2f}%")

    # Also measure OUTPUT MSE (Y = X @ W)
    Y_original = X @ W.T  # Shape: [batch, K]

    def compute_output_mse(mask):
        W_sparse_norm = W_norm * mask
        Q, scales, zeros, _ = quantize_rtn(W_sparse_norm, min_max, group_size=group_size)
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
        W_deq = W_deq_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0)

        Y_approx = X @ W_deq.T
        return ((Y_original - Y_approx) ** 2).mean().item()

    output_mse_std = compute_output_mse(mask_std)
    output_mse_inv = compute_output_mse(mask_inv)
    output_mse_wanda = compute_output_mse(mask_wanda)

    print(f"\n--- Output MSE (Y = X @ W) ---")
    print(f"Standard:    {output_mse_std:.6f}")
    print(f"Inverse:     {output_mse_inv:.6f}")
    print(f"Pure Wanda:  {output_mse_wanda:.6f}")

    output_improvement_inv = (output_mse_std - output_mse_inv) / output_mse_std * 100
    output_improvement_wanda = (output_mse_std - output_mse_wanda) / output_mse_std * 100

    print(f"\nInverse vs Standard (output MSE): {output_improvement_inv:+.2f}%")
    print(f"Wanda vs Standard (output MSE): {output_improvement_wanda:+.2f}%")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if improvement_inv > 5 or output_improvement_inv > 5:
        print(">>> Inverse μ weighting shows SIGNIFICANT improvement!")
        print(">>> Worth investigating as a hypothesis.")
    elif improvement_inv > 0:
        print(">>> Inverse μ weighting shows MARGINAL improvement.")
        print(">>> Might not be worth pursuing.")
    else:
        print(">>> Inverse μ weighting is WORSE than standard.")
        print(">>> The current approach is better.")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_inverse_importance()
