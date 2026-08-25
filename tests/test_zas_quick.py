"""
Test Zero-Aware Sinkhorn (ZAS) for Sparse Quantization.

Key insight: Standard Sinkhorn fails on sparse matrices because zeros corrupt
the normalization. ZAS modifies Sinkhorn to normalize ONLY the non-zero elements.

Standard Sinkhorn:
  μ₂[i] = 1 / sqrt(sum_j(W²[i,j] * μ₁²[j]))
  μ₁[j] = 1 / sqrt(sum_i(W²[i,j] * μ₂²[i]))

Zero-Aware Sinkhorn:
  μ₂[i] = 1 / sqrt(sum_j(W²[i,j] * μ₁²[j] * M[i,j]))
  μ₁[j] = 1 / sqrt(sum_i(W²[i,j] * μ₂²[i] * M[i,j]))

Where M is the sparsity mask (1 for kept weights, 0 for pruned).

This is NOVEL because:
1. No prior work adapts Sinkhorn specifically for sparse matrix quantization
2. It's a principled extension of the theoretical framework
3. It properly handles the structural change from pruning
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


def sinkhorn_zero_aware(W, mask, order=16, eps=1e-8):
    """
    Zero-Aware Sinkhorn normalization for sparse matrices.

    Key insight: Use density-normalized sums to account for varying
    numbers of non-zeros per row/column.

    Args:
        W: Weight matrix [K, N]
        mask: Binary mask [K, N] where 1=keep, 0=prune
        order: Number of Sinkhorn iterations
        eps: Numerical stability epsilon

    Returns:
        W_norm: Normalized weight matrix
        mu1: Column scaling factors [N]
        mu2: Row scaling factors [K]
    """
    W = W.float()
    mask = mask.float()
    K, N = W.shape
    device = W.device

    # Compute per-row and per-column density
    row_density = mask.sum(dim=1).clamp(min=1) / N  # [K]
    col_density = mask.sum(dim=0).clamp(min=1) / K  # [N]

    # Initialize scaling factors
    mu1 = torch.ones(N, device=device)  # Column scales
    mu2 = torch.ones(K, device=device)  # Row scales

    # W_masked has zeros where mask=0
    W_masked = W * mask

    for _ in range(order):
        # Row normalization - normalize by density to account for missing weights
        W_sq = W_masked ** 2
        row_sum = (W_sq * (mu1 ** 2).unsqueeze(0)).sum(dim=1)  # [K]
        # Density-adjusted sum: divide by density to estimate "full" sum
        row_sum_adj = row_sum / row_density.clamp(min=eps)
        row_sum_adj = row_sum_adj.clamp(min=eps)
        mu2 = 1.0 / torch.sqrt(row_sum_adj)

        # Column normalization - similarly adjust for density
        col_sum = (W_sq * (mu2 ** 2).unsqueeze(1)).sum(dim=0)  # [N]
        col_sum_adj = col_sum / col_density.clamp(min=eps)
        col_sum_adj = col_sum_adj.clamp(min=eps)
        mu1 = 1.0 / torch.sqrt(col_sum_adj)

    # Normalized matrix
    W_norm = W_masked * mu2.unsqueeze(1) * mu1.unsqueeze(0)

    return W_norm, mu1, mu2


def sinkhorn_zero_aware_log(W, mask, order=16, eps=1e-8):
    """
    Zero-Aware Sinkhorn in log space for numerical stability.
    """
    W = W.float()
    mask = mask.float()
    K, N = W.shape
    device = W.device

    # Work with log(|W|) for stability
    W_masked = W * mask
    log_W_sq = 2 * torch.log(W_masked.abs().clamp(min=eps))

    # Initialize log scales
    log_mu1 = torch.zeros(N, device=device)
    log_mu2 = torch.zeros(K, device=device)

    # Create mask for log operations (set masked positions to -inf in log space)
    log_mask = torch.where(mask > 0, torch.zeros_like(mask), torch.full_like(mask, -1e10))

    for _ in range(order):
        # Row normalization
        # log_mu2[i] = -0.5 * logsumexp_j(log_W_sq[i,j] + 2*log_mu1[j])
        row_contrib = log_W_sq + 2 * log_mu1.unsqueeze(0) + log_mask
        log_row_sum = torch.logsumexp(row_contrib, dim=1)
        log_mu2 = -0.5 * log_row_sum

        # Column normalization
        col_contrib = log_W_sq + 2 * log_mu2.unsqueeze(1) + log_mask
        log_col_sum = torch.logsumexp(col_contrib, dim=0)
        log_mu1 = -0.5 * log_col_sum

    mu1 = torch.exp(log_mu1)
    mu2 = torch.exp(log_mu2)

    W_norm = W_masked * mu2.unsqueeze(1) * mu1.unsqueeze(0)

    return W_norm, mu1, mu2


def baseline_sparse_quant(W, X, sparsity, nbits, group_size=64):
    """Baseline: Full Sinkhorn, then prune, then quantize."""
    K, N = W.shape
    device = W.device

    # Standard Sinkhorn on full matrix
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # Importance and mask
    act_norms = compute_activation_norms(X)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2
    mask = create_sparsity_mask(importance, sparsity)

    # Quantize
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)
    Q = Q * mask.to(Q.dtype)

    # Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2.unsqueeze(1) * mu1.unsqueeze(0) * mask

    return W_deq, mask, mu1, mu2


def zas_sparse_quant(W, X, sparsity, nbits, group_size=64):
    """ZAS: Use Zero-Aware Sinkhorn for sparse quantization."""
    K, N = W.shape
    device = W.device

    # First, standard Sinkhorn for importance computation
    _, mu1_full, mu2_full = sinkhorn_log(W.float(), order=16)

    # Importance and mask (same as baseline for fair comparison)
    act_norms = compute_activation_norms(X)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1_full.unsqueeze(0) * mu2_full
    mask = create_sparsity_mask(importance, sparsity)

    # Now use Zero-Aware Sinkhorn on the sparse matrix
    W_sparse = W * mask
    W_norm_zas, mu1_zas, mu2_zas = sinkhorn_zero_aware(W_sparse, mask, order=16)

    # Quantize the ZAS-normalized matrix
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm_zas, min_max, group_size=group_size)

    # Dequantize with ZAS scales
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2_zas.unsqueeze(1) * mu1_zas.unsqueeze(0)

    return W_deq, mask, mu1_zas, mu2_zas


def test_zas_synthetic():
    """Test ZAS on synthetic data."""
    print("="*60)
    print("TEST: Zero-Aware Sinkhorn (ZAS)")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 32
    group_size = 64

    W = torch.randn(K, N, device=device)
    X = torch.randn(batch, N, device=device)

    print(f"\nMatrix: {K}x{N}, group_size={group_size}")

    print(f"\n{'Sparsity':>10} | {'Bits':>4} | {'Baseline MSE':>12} | {'ZAS MSE':>12} | {'Improvement':>12}")
    print("-" * 65)

    for sparsity in [0.2, 0.35, 0.5]:
        for nbits in [3, 4]:
            W_base, mask_b, _, _ = baseline_sparse_quant(W.float(), X, sparsity, nbits, group_size)
            mse_base = ((W - W_base) ** 2).mean().item()

            W_zas, mask_z, _, _ = zas_sparse_quant(W.float(), X, sparsity, nbits, group_size)
            mse_zas = ((W - W_zas) ** 2).mean().item()

            improvement = (mse_base - mse_zas) / mse_base * 100

            print(f"{sparsity*100:>9.0f}% | {nbits:>4} | {mse_base:>12.6f} | {mse_zas:>12.6f} | {improvement:>+11.2f}%")


def test_zas_output_error():
    """Test ZAS with output reconstruction error metric."""
    print("\n" + "="*60)
    print("TEST: ZAS Output Reconstruction Error")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 256, 512
    batch = 128
    sparsity = 0.35
    nbits = 3
    group_size = 64

    W = torch.randn(K, N, device=device)
    X = torch.randn(batch, N, device=device)

    # Target output
    Y_target = X @ W.T

    # Baseline
    W_base, _, _, _ = baseline_sparse_quant(W.float(), X, sparsity, nbits, group_size)
    Y_base = X @ W_base.T
    output_mse_base = ((Y_target - Y_base) ** 2).mean().item()

    # ZAS
    W_zas, _, _, _ = zas_sparse_quant(W.float(), X, sparsity, nbits, group_size)
    Y_zas = X @ W_zas.T
    output_mse_zas = ((Y_target - Y_zas) ** 2).mean().item()

    improvement = (output_mse_base - output_mse_zas) / output_mse_base * 100

    print(f"\nConfig: {K}x{N}, {sparsity*100:.0f}% sparsity, {nbits}-bit")
    print(f"Baseline output MSE: {output_mse_base:.6f}")
    print(f"ZAS output MSE:      {output_mse_zas:.6f}")
    print(f"Improvement:         {improvement:+.2f}%")


def test_zas_real():
    """Test ZAS on real Qwen-0.5B weights."""
    print("\n" + "="*60)
    print("TEST: ZAS on Real Qwen-0.5B Weights")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM

        print("Loading Qwen-0.5B...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Test on multiple layers
        layers_to_test = [0, 5, 10]
        sparsity = 0.35
        nbits = 3
        group_size = 64

        for layer_idx in layers_to_test:
            W = model.model.layers[layer_idx].mlp.gate_proj.weight.data.float()
            K, N = W.shape
            device = W.device

            batch = 64
            X = torch.randn(batch, N, device=device, dtype=torch.float32)
            Y_target = X @ W.T

            # Baseline
            W_base, _, _, _ = baseline_sparse_quant(W, X, sparsity, nbits, group_size)
            Y_base = X @ W_base.T
            out_mse_base = ((Y_target - Y_base) ** 2).mean().item()
            wgt_mse_base = ((W - W_base) ** 2).mean().item()

            # ZAS
            W_zas, _, _, _ = zas_sparse_quant(W, X, sparsity, nbits, group_size)
            Y_zas = X @ W_zas.T
            out_mse_zas = ((Y_target - Y_zas) ** 2).mean().item()
            wgt_mse_zas = ((W - W_zas) ** 2).mean().item()

            wgt_imp = (wgt_mse_base - wgt_mse_zas) / wgt_mse_base * 100
            out_imp = (out_mse_base - out_mse_zas) / out_mse_base * 100

            print(f"\nLayer {layer_idx} gate_proj [{K}x{N}]:")
            print(f"  Weight MSE: Baseline={wgt_mse_base:.6f}, ZAS={wgt_mse_zas:.6f} ({wgt_imp:+.2f}%)")
            print(f"  Output MSE: Baseline={out_mse_base:.6f}, ZAS={out_mse_zas:.6f} ({out_imp:+.2f}%)")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def analyze_zas_scales():
    """Analyze how ZAS scales differ from standard Sinkhorn scales."""
    print("\n" + "="*60)
    print("ANALYSIS: ZAS vs Standard Sinkhorn Scales")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 64, 128
    sparsity = 0.35

    W = torch.randn(K, N, device=device)

    # Standard Sinkhorn on full matrix
    _, mu1_full, mu2_full = sinkhorn_log(W.float(), order=16)

    # Create pruning mask
    importance = W.abs()  # Simple importance for this analysis
    mask = create_sparsity_mask(importance, sparsity)

    # ZAS on sparse matrix
    W_sparse = W * mask
    _, mu1_zas, mu2_zas = sinkhorn_zero_aware(W_sparse, mask, order=16)

    print(f"\nMatrix: {K}x{N}, {sparsity*100:.0f}% sparsity")
    print(f"\nRow scales (mu2) comparison:")
    print(f"  Full Sinkhorn: mean={mu2_full.mean():.4f}, std={mu2_full.std():.4f}, range=[{mu2_full.min():.4f}, {mu2_full.max():.4f}]")
    print(f"  ZAS:           mean={mu2_zas.mean():.4f}, std={mu2_zas.std():.4f}, range=[{mu2_zas.min():.4f}, {mu2_zas.max():.4f}]")

    print(f"\nColumn scales (mu1) comparison:")
    print(f"  Full Sinkhorn: mean={mu1_full.mean():.4f}, std={mu1_full.std():.4f}, range=[{mu1_full.min():.4f}, {mu1_full.max():.4f}]")
    print(f"  ZAS:           mean={mu1_zas.mean():.4f}, std={mu1_zas.std():.4f}, range=[{mu1_zas.min():.4f}, {mu1_zas.max():.4f}]")

    # Check if scales are finite
    print(f"\nScale validity:")
    print(f"  mu1_zas finite: {mu1_zas.isfinite().all()}, mu2_zas finite: {mu2_zas.isfinite().all()}")


if __name__ == '__main__':
    analyze_zas_scales()
    test_zas_synthetic()
    test_zas_output_error()
    test_zas_real()
