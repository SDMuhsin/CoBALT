"""
Quick test: Quantize-First Sparse (QFS)

Current pipeline: Prune → Quantize
QFS pipeline: Quantize → Prune based on quantized values

Idea: Some weights might have high importance but round to bad quantized values.
By quantizing first, we can prune based on ACTUAL quantized representation.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def test_qfs():
    print("="*70)
    print("QUANTIZE-FIRST SPARSE (QFS) QUICK TEST")
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

    # Quantize (4-bit)
    nbits = 4
    group_size = 64
    min_max = [0, 2**nbits - 1]

    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    # Dequantize to get W_q (quantized values in normalized space)
    n_groups = scales.shape[1]
    Q_grouped = Q.view(K, n_groups, group_size)
    W_q_norm = (Q_grouped - zeros) * scales
    W_q_norm = W_q_norm.view(K, N)

    # Dequantize to original space
    W_q = W_q_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0)

    # Method 1: Standard (prune original, then quantize)
    # Inverse μ importance on original weights
    importance_std = W.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # Method 2: QFS (quantize first, prune based on quantized)
    # Importance based on quantized weights
    importance_qfs = W_q.abs() * act_norms.unsqueeze(0) / (mu1.unsqueeze(0) * mu2.unsqueeze(1) + 1e-6)

    # Method 3: QFS with quantization error awareness
    # Importance = quantized_importance × (1 - relative_error)
    quant_error = (W - W_q).abs()
    relative_error = quant_error / (W.abs() + 1e-6)
    importance_qfs_aware = importance_qfs * (1 - relative_error.clamp(max=0.9))

    # Create masks (35% sparsity)
    sparsity = 0.35
    n_prune = int(K * N * sparsity)

    def create_mask(importance):
        threshold = importance.view(-1).sort().values[n_prune]
        return (importance > threshold).float()

    mask_std = create_mask(importance_std)
    mask_qfs = create_mask(importance_qfs)
    mask_qfs_aware = create_mask(importance_qfs_aware)

    # Measure reconstruction MSE
    # For standard: apply mask to W_norm, then quantize
    W_sparse_norm_std = W_norm * mask_std
    Q_std, s_std, z_std, _ = quantize_rtn(W_sparse_norm_std, min_max, group_size=group_size)
    Q_grouped_std = Q_std.view(K, n_groups, group_size)
    W_deq_norm_std = (Q_grouped_std - z_std) * s_std
    W_deq_std = W_deq_norm_std.view(K, N) * mu2.unsqueeze(1) * mu1.unsqueeze(0)

    # For QFS methods: apply mask to already-quantized W_q
    W_deq_qfs = W_q * mask_qfs
    W_deq_qfs_aware = W_q * mask_qfs_aware

    # Compute MSE (on non-zero weights)
    W_masked_std = W * mask_std
    W_masked_qfs = W * mask_qfs
    W_masked_qfs_aware = W * mask_qfs_aware

    mse_std = ((W_masked_std - W_deq_std * mask_std) ** 2).sum() / mask_std.sum()
    mse_qfs = ((W_masked_qfs - W_deq_qfs) ** 2).sum() / mask_qfs.sum()
    mse_qfs_aware = ((W_masked_qfs_aware - W_deq_qfs_aware) ** 2).sum() / mask_qfs_aware.sum()

    print(f"\n--- Reconstruction MSE ---")
    print(f"Standard (prune→quant): {mse_std:.6f}")
    print(f"QFS (quant→prune):      {mse_qfs:.6f}")
    print(f"QFS-Aware:              {mse_qfs_aware:.6f}")

    improvement_qfs = (mse_std - mse_qfs) / mse_std * 100
    improvement_qfs_aware = (mse_std - mse_qfs_aware) / mse_std * 100

    print(f"\nQFS vs Standard: {improvement_qfs:+.2f}%")
    print(f"QFS-Aware vs Standard: {improvement_qfs_aware:+.2f}%")

    # Check mask overlap
    overlap_qfs = (mask_std * mask_qfs).sum() / mask_std.sum()
    overlap_qfs_aware = (mask_std * mask_qfs_aware).sum() / mask_std.sum()

    print(f"\nMask overlap with Standard:")
    print(f"  QFS: {overlap_qfs:.1%}")
    print(f"  QFS-Aware: {overlap_qfs_aware:.1%}")

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    test_qfs()
