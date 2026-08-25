"""
Quick test: Benefit/Cost Pruning (BCP)

Idea: Frame pruning as benefit/cost trade-off:
- Benefit: importance = |W| × ||X||
- Cost: predicted quant error = μ₁ × μ₂ × scale

Score = benefit / cost = |W| × ||X|| / (μ₁ × μ₂ × scale)

This goes beyond inverse μ by incorporating per-group quantization scale.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_bcp():
    print("="*70)
    print("BENEFIT/COST PRUNING (BCP) QUICK TEST")
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

    # Create synthetic activations
    batch = 64
    X = torch.randn(batch, N, device=W.device, dtype=W.dtype) * 0.1
    act_norms = torch.norm(X, dim=0)

    # Compute Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W, order=16)
    mu2 = mu2.squeeze()

    # First quantize to get scales
    nbits = 4
    group_size = 64
    min_max = [0, 2**nbits - 1]

    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    # Expand scales to per-weight
    n_groups = scales.shape[1]
    scales_expanded = scales.squeeze(-1).repeat_interleave(group_size, dim=1)  # [K, N]

    # Method 1: Standard inverse μ
    importance_inv_mu = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # Method 2: BCP (benefit/cost with scale)
    mu_product = mu1.unsqueeze(0) * mu2.unsqueeze(1)
    cost = mu_product * scales_expanded
    importance_bcp = W.abs() * act_norms.unsqueeze(0) / (cost + 1e-6)

    # Method 3: Scale-only (ignore μ, just use scale)
    importance_scale = W.abs() * act_norms.unsqueeze(0) / (scales_expanded + 1e-6)

    # Create masks (35% sparsity)
    sparsity = 0.35
    n_prune = int(K * N * sparsity)

    def create_mask(importance):
        threshold = importance.view(-1).sort().values[n_prune]
        return (importance > threshold).float()

    mask_inv_mu = create_mask(importance_inv_mu)
    mask_bcp = create_mask(importance_bcp)
    mask_scale = create_mask(importance_scale)

    # Apply masks and quantize
    def evaluate_mask(mask, name):
        W_sparse = W_norm * mask
        Q, s, z, _ = quantize_rtn(W_sparse, min_max, group_size=group_size)
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - z) * s
        W_deq = W_deq_norm.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)

        # MSE on non-zero weights
        W_masked = W * mask
        mse = ((W_masked - W_deq * mask) ** 2).sum() / mask.sum()
        return mse.item()

    mse_inv_mu = evaluate_mask(mask_inv_mu, "Inverse μ")
    mse_bcp = evaluate_mask(mask_bcp, "BCP")
    mse_scale = evaluate_mask(mask_scale, "Scale-only")

    print(f"\n--- Reconstruction MSE ---")
    print(f"Inverse μ (baseline):  {mse_inv_mu:.6f}")
    print(f"BCP (μ + scale):       {mse_bcp:.6f} ({(mse_inv_mu - mse_bcp) / mse_inv_mu * 100:+.2f}%)")
    print(f"Scale-only:            {mse_scale:.6f} ({(mse_inv_mu - mse_scale) / mse_inv_mu * 100:+.2f}%)")

    # Check mask overlap
    overlap_bcp = (mask_inv_mu * mask_bcp).sum() / mask_inv_mu.sum()
    overlap_scale = (mask_inv_mu * mask_scale).sum() / mask_inv_mu.sum()

    print(f"\nMask overlap with Inverse μ:")
    print(f"  BCP: {overlap_bcp:.1%}")
    print(f"  Scale-only: {overlap_scale:.1%}")

    # Analyze scale variation
    print(f"\n--- Scale Statistics ---")
    print(f"Scale mean: {scales.mean():.6f}")
    print(f"Scale std:  {scales.std():.6f}")
    print(f"Scale CV:   {scales.std() / scales.mean():.2%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_bcp()
