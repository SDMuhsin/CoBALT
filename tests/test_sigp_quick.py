"""
Test Sinkhorn Imbalance-Guided Pruning (SIGP).

Novel insight: Sinkhorn normalization aims to balance row/column variances.
Some weights might be "outliers" that make balancing harder.
Removing them could IMPROVE Sinkhorn normalization quality.

The imbalance metric is: max(std) / min(std) across rows/columns.
Lower imbalance = better balanced matrix = potentially better quantization.

SIGP approach:
1. For each weight, estimate how its removal affects Sinkhorn imbalance
2. Prune weights that REDUCE imbalance (help balance)
3. This prioritizes weights that obstruct normalization

This is novel because:
- Existing methods prune based on magnitude/activation importance
- SIGP prunes based on impact on normalization QUALITY
- It leverages Sinkhorn's unique optimization objective
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn
from sinq.sparse_quant import compute_activation_norms


def compute_imbalance(W):
    """Compute Sinkhorn-style imbalance metric."""
    row_std = W.std(dim=1)
    col_std = W.std(dim=0)

    all_std = torch.cat([row_std, col_std])
    s_min = all_std.min().clamp(min=1e-8)
    s_max = all_std.max()

    return (s_max / s_min).item()


def estimate_imbalance_impact(W, i, j):
    """
    Estimate how removing weight W[i,j] affects imbalance.

    Returns: change in imbalance (negative = improvement)
    """
    # This is expensive to compute exactly for each weight
    # Use an approximation based on how much this weight deviates from row/col mean
    row_mean = W[i, :].mean()
    col_mean = W[:, j].mean()
    row_std = W[i, :].std()
    col_std = W[:, j].std()

    # How much does this weight contribute to row/col variance?
    row_deviation = (W[i, j] - row_mean).abs() / (row_std + 1e-8)
    col_deviation = (W[i, j] - col_mean).abs() / (col_std + 1e-8)

    # High deviation = removing this weight reduces variance = potentially helps balance
    impact = row_deviation + col_deviation

    return impact


def sigp_importance(W, X, mu1, mu2, alpha=1.0):
    """
    Compute SIGP importance score.

    importance = wanda_importance - alpha * imbalance_impact

    Lower importance = better to prune
    Weights that help balance (high impact) get lower importance
    """
    K, N = W.shape
    device = W.device

    # Standard Wanda importance
    act_norms = torch.norm(X.float(), dim=0)
    wanda_imp = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2

    # Compute imbalance impact efficiently
    W_norm = W / (mu2 * mu1)

    # Row and column means/stds
    row_mean = W_norm.mean(dim=1, keepdim=True)  # [K, 1]
    col_mean = W_norm.mean(dim=0, keepdim=True)  # [1, N]
    row_std = W_norm.std(dim=1, keepdim=True).clamp(min=1e-8)  # [K, 1]
    col_std = W_norm.std(dim=0, keepdim=True).clamp(min=1e-8)  # [1, N]

    # Deviation from mean (proxy for impact on variance)
    row_dev = (W_norm - row_mean).abs() / row_std
    col_dev = (W_norm - col_mean).abs() / col_std

    imbalance_impact = row_dev + col_dev

    # SIGP importance: penalize weights that help balance (make them more likely to keep)
    # Or boost importance of weights that HURT balance (make them easier to prune)
    # We want to KEEP weights that help balance, PRUNE weights that hurt
    # So: importance = wanda - alpha * impact (high impact = helps balance = higher importance = keep)
    # Actually, high deviation means the weight IS an outlier, removing it helps
    # So: importance = wanda - alpha * deviation (high deviation = outlier = lower importance = prune)

    sigp_imp = wanda_imp - alpha * imbalance_impact * wanda_imp.mean()

    return sigp_imp


def create_mask(importance, sparsity):
    """Create pruning mask."""
    K, N = importance.shape
    n_prune = int(K * N * sparsity)
    flat = importance.view(-1)
    _, prune_idx = torch.topk(flat, n_prune, largest=False)
    mask = torch.ones(K * N, device=importance.device)
    mask[prune_idx] = 0
    return mask.view(K, N)


def baseline_sparse_quant(W, X, sparsity, nbits, group_size=64):
    """Baseline SINQ-Sparse."""
    K, N = W.shape

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # Wanda importance
    act_norms = torch.norm(X.float(), dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2

    # Mask and quantize
    mask = create_mask(importance, sparsity)

    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)
    Q = Q * mask.to(Q.dtype)

    # Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq = (Q_grouped - zeros) * scales
        W_deq = W_deq.view(K, N)
    else:
        W_deq = (Q - zeros) * scales

    W_deq = W_deq * mu2 * mu1 * mask

    return W_deq, mask, W_norm


def sigp_sparse_quant(W, X, sparsity, nbits, group_size=64, alpha=1.0):
    """SIGP SINQ-Sparse."""
    K, N = W.shape

    # Sinkhorn
    W_norm, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # SIGP importance
    importance = sigp_importance(W.float(), X, mu1, mu2, alpha=alpha)

    # Mask and quantize
    mask = create_mask(importance, sparsity)

    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm, min_max, group_size=group_size)
    Q = Q * mask.to(Q.dtype)

    # Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq = (Q_grouped - zeros) * scales
        W_deq = W_deq.view(K, N)
    else:
        W_deq = (Q - zeros) * scales

    W_deq = W_deq * mu2 * mu1 * mask

    # Check imbalance of sparse matrix
    imb_after = compute_imbalance(W * mask)

    return W_deq, mask, imb_after


def test_sigp():
    """Test SIGP approach."""
    print("="*60)
    print("TEST: Sinkhorn Imbalance-Guided Pruning (SIGP)")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 32
    sparsity = 0.35
    nbits = 3
    group_size = 64

    W = torch.randn(K, N, device=device)
    X = torch.randn(batch, N, device=device)

    # Initial imbalance
    imb_initial = compute_imbalance(W)
    print(f"\nInitial imbalance: {imb_initial:.2f}")

    # Baseline
    W_base, mask_base, W_norm = baseline_sparse_quant(W.float(), X, sparsity, nbits, group_size)
    mse_base = ((W - W_base) ** 2).mean().item()
    imb_base = compute_imbalance(W * mask_base)
    print(f"\nBaseline:")
    print(f"  MSE: {mse_base:.6f}")
    print(f"  Imbalance after prune: {imb_base:.2f}")

    # SIGP with different alpha
    print(f"\nSIGP results:")
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        W_sigp, mask_sigp, imb_sigp = sigp_sparse_quant(W.float(), X, sparsity, nbits, group_size, alpha=alpha)
        mse_sigp = ((W - W_sigp) ** 2).mean().item()
        improvement = (mse_base - mse_sigp) / mse_base * 100
        print(f"  alpha={alpha}: MSE={mse_sigp:.6f} ({improvement:+.2f}%), imbalance={imb_sigp:.2f}")


def test_sigp_real():
    """Test SIGP on real weights."""
    print("\n" + "="*60)
    print("TEST: SIGP on Real Qwen-0.5B Weights")
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

        imb_initial = compute_imbalance(W)
        print(f"Initial imbalance: {imb_initial:.2f}")

        W_base, mask_base, _ = baseline_sparse_quant(W, X, sparsity, nbits, group_size)
        mse_base = ((W - W_base) ** 2).mean().item()
        imb_base = compute_imbalance(W * mask_base)

        print(f"\nBaseline MSE: {mse_base:.6f}, imbalance: {imb_base:.2f}")

        best_alpha = None
        best_mse = mse_base
        for alpha in [0.1, 0.5, 1.0, 2.0, 5.0]:
            W_sigp, mask_sigp, imb_sigp = sigp_sparse_quant(W, X, sparsity, nbits, group_size, alpha=alpha)
            mse_sigp = ((W - W_sigp) ** 2).mean().item()
            improvement = (mse_base - mse_sigp) / mse_base * 100
            print(f"SIGP alpha={alpha}: MSE={mse_sigp:.6f} ({improvement:+.2f}%), imbalance={imb_sigp:.2f}")
            if mse_sigp < best_mse:
                best_mse = mse_sigp
                best_alpha = alpha

        if best_alpha:
            print(f"\nBest alpha: {best_alpha} with MSE improvement: {(mse_base - best_mse) / mse_base * 100:+.2f}%")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_sigp()
    test_sigp_real()
