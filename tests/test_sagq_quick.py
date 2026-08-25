"""
Test Sparse-Aware Group Quantization (SAGQ).

Current baseline:
1. Sinkhorn on full matrix → μ₁, μ₂
2. Prune and create mask
3. Quantize FULL W_norm (including to-be-pruned weights)
4. Apply mask to quantized weights

Problem: Quantization scales are computed using ALL weights including pruned ones.
Pruned weights (especially outliers) affect the scale unnecessarily.

SAGQ approach:
1. Sinkhorn on full matrix → μ₁, μ₂
2. Prune and create mask
3. Quantize W_norm with scales computed ONLY on non-zero weights
4. This optimizes scales for the actual kept weights

Key insight: The quantization scale per group should ignore zeros,
not include them in the min/max computation.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn.functional as F
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import (
    compute_activation_norms,
    create_sparsity_mask,
)


def quantize_rtn_sparse_aware(matrix, mask, min_max, group_size=64):
    """
    RTN quantization with scales computed only on non-zero (kept) weights.

    For each group, compute scale based on min/max of KEPT weights only.
    """
    w = matrix.float()
    K, N = w.shape
    device = w.device
    orig_dtype = matrix.dtype

    max_int = min_max[1]
    min_int = min_max[0]

    # Handle group-wise quantization
    use_groups = group_size is not None and group_size > 0 and N > group_size

    if use_groups:
        # Pad if needed
        if N % group_size != 0:
            pad_size = group_size - (N % group_size)
            w = F.pad(w, (0, pad_size), value=0)
            mask_padded = F.pad(mask, (0, pad_size), value=0)
            N_padded = w.shape[1]
        else:
            N_padded = N
            pad_size = 0
            mask_padded = mask

        n_groups = N_padded // group_size

        # Reshape to [K * n_groups, group_size]
        w_grouped = w.view(K, n_groups, group_size).reshape(K * n_groups, group_size)
        mask_grouped = mask_padded.view(K, n_groups, group_size).reshape(K * n_groups, group_size)

        # For each group, compute min/max only on non-zero elements
        # Use large positive/negative values for zeros to exclude them
        w_for_max = w_grouped.clone()
        w_for_max[mask_grouped == 0] = -1e10  # Will not be max
        max_val = w_for_max.amax(dim=1, keepdim=True)

        w_for_min = w_grouped.clone()
        w_for_min[mask_grouped == 0] = 1e10  # Will not be min
        min_val = w_for_min.amin(dim=1, keepdim=True)

        # Handle groups that are fully masked (all zeros)
        fully_masked = (mask_grouped.sum(dim=1, keepdim=True) == 0)
        max_val = torch.where(fully_masked, torch.ones_like(max_val), max_val)
        min_val = torch.where(fully_masked, torch.zeros_like(min_val), min_val)

        # Compute scales
        scales = (max_val - min_val).clamp(min=1e-4) / max_int
        zeros = -torch.round(min_val / scales)

        # Quantize
        q = torch.clamp(torch.round(w_grouped / scales + zeros), min_int, max_int)
        q = q.to(torch.int8)

        # Reshape back
        q = q.view(K, n_groups, group_size).view(K, N_padded)
        if pad_size > 0:
            q = q[:, :N]
        scales = scales.view(K, n_groups, 1)
        zeros = zeros.view(K, n_groups, 1)

    else:
        # Per-row quantization with sparse-aware min/max
        w_for_max = w.clone()
        w_for_max[mask == 0] = -1e10
        max_val = w_for_max.amax(dim=1, keepdim=True)

        w_for_min = w.clone()
        w_for_min[mask == 0] = 1e10
        min_val = w_for_min.amin(dim=1, keepdim=True)

        fully_masked = (mask.sum(dim=1, keepdim=True) == 0)
        max_val = torch.where(fully_masked, torch.ones_like(max_val), max_val)
        min_val = torch.where(fully_masked, torch.zeros_like(min_val), min_val)

        scales = (max_val - min_val).clamp(min=1e-4) / max_int
        zeros = -torch.round(min_val / scales)
        q = torch.clamp(torch.round(w / scales + zeros), min_int, max_int)
        q = q.to(torch.int8)

    return q.contiguous(), scales.to(orig_dtype), zeros.to(orig_dtype), orig_dtype


def baseline_sparse_quantize(W, X, sparsity, nbits, group_size=64):
    """Baseline: quantize full matrix, then apply mask."""
    K, N = W.shape
    device = W.device

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # Importance and mask
    act_norms = torch.norm(X.float(), dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2
    mask = create_sparsity_mask(importance, sparsity)

    # Quantize FULL matrix
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)

    # Apply mask after quantization
    Q = Q * mask.to(Q.dtype)

    # Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2 * mu1 * mask

    return W_deq, mask


def sagq_sparse_quantize(W, X, sparsity, nbits, group_size=64):
    """SAGQ: Compute quantization scales only on non-zero weights."""
    K, N = W.shape
    device = W.device

    # Sinkhorn (same as baseline)
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # Importance and mask (same as baseline)
    act_norms = torch.norm(X.float(), dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2
    mask = create_sparsity_mask(importance, sparsity)

    # Quantize with SPARSE-AWARE scales
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn_sparse_aware(W_norm, mask, min_max, group_size=group_size)

    # Apply mask
    Q = Q * mask.to(Q.dtype)

    # Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2 * mu1 * mask

    return W_deq, mask


def test_sagq():
    """Test SAGQ approach."""
    print("="*60)
    print("TEST: Sparse-Aware Group Quantization (SAGQ)")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 32
    group_size = 64

    W = torch.randn(K, N, device=device)
    X = torch.randn(batch, N, device=device)

    print(f"\nMatrix: {K}x{N}, group_size={group_size}")

    print(f"\n{'Sparsity':>10} | {'Bits':>4} | {'Baseline MSE':>12} | {'SAGQ MSE':>12} | {'Improvement':>12}")
    print("-" * 65)

    for sparsity in [0.2, 0.35, 0.5]:
        for nbits in [3, 4]:
            W_baseline, mask_b = baseline_sparse_quantize(W.float(), X, sparsity, nbits, group_size)
            mse_baseline = ((W - W_baseline) ** 2).mean().item()

            W_sagq, mask_s = sagq_sparse_quantize(W.float(), X, sparsity, nbits, group_size)
            mse_sagq = ((W - W_sagq) ** 2).mean().item()

            improvement = (mse_baseline - mse_sagq) / mse_baseline * 100

            print(f"{sparsity*100:>9.0f}% | {nbits:>4} | {mse_baseline:>12.6f} | {mse_sagq:>12.6f} | {improvement:>+11.2f}%")


def test_sagq_real():
    """Test SAGQ on real weights."""
    print("\n" + "="*60)
    print("TEST: SAGQ on Real Qwen-0.5B Weights")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM

        print("Loading Qwen-0.5B...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        W = model.model.layers[0].mlp.gate_proj.weight.data.float()
        K, N = W.shape
        device = W.device

        print(f"\nLayer 0 gate_proj: {K}x{N}")

        batch = 64
        X = torch.randn(batch, N, device=device, dtype=torch.float32)

        sparsity = 0.35
        nbits = 3
        group_size = 64

        W_baseline, _ = baseline_sparse_quantize(W, X, sparsity, nbits, group_size)
        mse_baseline = ((W - W_baseline) ** 2).mean().item()

        W_sagq, _ = sagq_sparse_quantize(W, X, sparsity, nbits, group_size)
        mse_sagq = ((W - W_sagq) ** 2).mean().item()

        improvement = (mse_baseline - mse_sagq) / mse_baseline * 100

        print(f"\nResults at {sparsity*100:.0f}% sparsity, {nbits}-bit:")
        print(f"Baseline MSE: {mse_baseline:.6f}")
        print(f"SAGQ MSE:     {mse_sagq:.6f}")
        print(f"Improvement:  {improvement:+.2f}%")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_sagq()
    test_sagq_real()
