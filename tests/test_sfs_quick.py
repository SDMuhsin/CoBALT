"""
Test Sparse-First Sinkhorn (SFS) hypothesis.

Current pipeline (baseline):
1. Sinkhorn on W → μ₁, μ₂ (computed on FULL matrix)
2. Prune based on importance
3. Quantize W_norm × mask
4. Dequantize with original μ₁, μ₂

Proposed SFS pipeline:
1. Prune first (based on Wanda importance on original W)
2. Sinkhorn on W × mask → μ₁', μ₂' (computed on SPARSE matrix!)
3. Quantize W_norm_sparse
4. Dequantize with μ₁', μ₂'

Hypothesis: Sinkhorn factors computed on sparse matrix are better suited
for quantizing the sparse matrix.
"""

import sys
sys.path.insert(0, '/workspace/SINQ')

import torch
import torch.nn.functional as F
from sinq.sinkhorn import sinkhorn_log
from sinq.dual_shift import quantize_rtn


def create_wanda_mask(W, X, sparsity):
    """Create Wanda-style pruning mask."""
    K, N = W.shape
    act_norms = torch.norm(X.float(), dim=0)  # [N]
    importance = W.abs() * act_norms.unsqueeze(0)

    n_prune = int(K * N * sparsity)
    flat_importance = importance.view(-1)
    _, prune_idx = torch.topk(flat_importance, n_prune, largest=False)

    mask = torch.ones(K * N, device=W.device)
    mask[prune_idx] = 0
    return mask.view(K, N)


def sinkhorn_with_mask(W, mask, order=16):
    """
    Run Sinkhorn normalization on sparse matrix.

    Key insight: We want to balance the variance of KEPT weights only.
    Zero weights should not affect the scaling computation.

    Approach: Replace zeros with the row/column mean during Sinkhorn,
    then apply factors to original sparse matrix.
    """
    K, N = W.shape
    W_sparse = W * mask

    # Option 1: Simple - just run Sinkhorn on sparse matrix
    # Problem: zeros will dominate variance computation

    # Option 2: Replace zeros with row mean before Sinkhorn
    # Then apply the computed factors to original sparse matrix
    row_sums = W_sparse.sum(dim=1, keepdim=True)
    row_counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
    row_means = row_sums / row_counts

    W_filled = W_sparse.clone()
    W_filled[mask == 0] = 0  # Keep zeros as zeros for Sinkhorn

    # Run Sinkhorn - it will handle zeros by considering variance of non-zero elements
    # Actually, let's use a modified approach: scale factors based on non-zero elements
    W_norm, mu1, mu2 = sinkhorn_log(W_sparse, order=order)

    return W_norm, mu1, mu2


def baseline_sinq_sparse(W, X, sparsity, nbits, group_size=64):
    """
    Baseline: Sinkhorn on full matrix, then prune, then quantize.
    """
    K, N = W.shape
    device = W.device

    # 1. Sinkhorn on FULL matrix
    W_norm_full, mu1, mu2 = sinkhorn_log(W.float(), order=16)

    # 2. Create mask based on importance (uses full matrix info)
    act_norms = torch.norm(X.float(), dim=0)
    importance = W.abs() * act_norms.unsqueeze(0) * mu1.unsqueeze(0) * mu2
    mask = create_wanda_mask_from_importance(importance, sparsity)

    # 3. Apply mask to normalized weights
    W_norm_sparse = W_norm_full * mask

    # 4. Quantize
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm_sparse, min_max, group_size=group_size)

    # 5. Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2 * mu1 * mask

    return W_deq, mask


def create_wanda_mask_from_importance(importance, sparsity):
    """Create mask from pre-computed importance scores."""
    K, N = importance.shape
    n_prune = int(K * N * sparsity)
    flat_importance = importance.view(-1)
    _, prune_idx = torch.topk(flat_importance, n_prune, largest=False)

    mask = torch.ones(K * N, device=importance.device)
    mask[prune_idx] = 0
    return mask.view(K, N)


def sfs_sinq_sparse(W, X, sparsity, nbits, group_size=64):
    """
    Sparse-First Sinkhorn: Prune first, then Sinkhorn on sparse matrix.
    """
    K, N = W.shape
    device = W.device

    # 1. Create mask based on simple Wanda importance (no Sinkhorn factors)
    mask = create_wanda_mask(W.float(), X, sparsity)

    # 2. Apply mask FIRST
    W_sparse = W * mask

    # 3. Sinkhorn on SPARSE matrix
    # Handle zeros: we need Sinkhorn to ignore zero elements
    # Simple approach: run Sinkhorn normally, zeros contribute to variance
    W_norm_sparse, mu1, mu2 = sinkhorn_log(W_sparse.float(), order=16)

    # 4. Quantize the Sinkhorn-normalized sparse matrix
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm_sparse, min_max, group_size=group_size)

    # 5. Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2 * mu1 * mask

    return W_deq, mask


def sfs_sinq_sparse_v2(W, X, sparsity, nbits, group_size=64):
    """
    Sparse-First Sinkhorn v2: Fill zeros with small values before Sinkhorn.

    The idea: zeros in sparse matrix dominate variance, making Sinkhorn
    compute suboptimal factors. Fill zeros with small noise, run Sinkhorn,
    then zero them out.
    """
    K, N = W.shape
    device = W.device

    # 1. Create mask based on simple Wanda importance
    mask = create_wanda_mask(W.float(), X, sparsity)

    # 2. For Sinkhorn, fill zeros with small values to not dominate variance
    W_sparse = W * mask

    # Fill zeros with row mean scaled down
    row_means = (W_sparse.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1)).abs()
    fill_value = row_means * 0.01  # Small fraction of row mean
    W_filled = W_sparse + (1 - mask) * fill_value

    # 3. Sinkhorn on filled matrix
    W_norm_filled, mu1, mu2 = sinkhorn_log(W_filled.float(), order=16)

    # 4. Apply mask to normalized weights (zero out the filled positions)
    # Recompute normalized sparse using original sparse and new factors
    W_norm_sparse = (W * mask) / (mu2 * mu1 + 1e-8)

    # 5. Quantize
    min_max = [0, 2**nbits - 1]
    Q, scales, zeros, _ = quantize_rtn(W_norm_sparse, min_max, group_size=group_size)

    # 6. Dequantize
    if len(scales.shape) == 3:
        n_groups = scales.shape[1]
        Q_grouped = Q.view(K, n_groups, group_size)
        W_deq_norm = (Q_grouped - zeros) * scales
        W_deq_norm = W_deq_norm.view(K, N)
    else:
        W_deq_norm = (Q - zeros) * scales

    W_deq = W_deq_norm * mu2 * mu1 * mask

    return W_deq, mask


def test_sfs():
    """Test Sparse-First Sinkhorn approaches."""
    print("="*60)
    print("TEST: Sparse-First Sinkhorn (SFS)")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 32
    sparsity = 0.35
    nbits = 3  # Test at 3-bit to match challenging baseline
    group_size = 64

    W = torch.randn(K, N, device=device)
    X = torch.randn(batch, N, device=device)

    print(f"\nConfig: {K}x{N}, {sparsity*100:.0f}% sparsity, {nbits}-bit")

    # Baseline
    W_baseline, mask_baseline = baseline_sinq_sparse(W, X, sparsity, nbits, group_size)
    mse_baseline = ((W - W_baseline) ** 2).mean().item()
    print(f"\nBaseline MSE: {mse_baseline:.6f}")

    # SFS v1: Simple Sinkhorn on sparse matrix
    W_sfs1, mask_sfs1 = sfs_sinq_sparse(W, X, sparsity, nbits, group_size)
    mse_sfs1 = ((W - W_sfs1) ** 2).mean().item()
    improvement1 = (mse_baseline - mse_sfs1) / mse_baseline * 100
    print(f"SFS v1 MSE:   {mse_sfs1:.6f} ({improvement1:+.2f}%)")

    # SFS v2: Fill zeros before Sinkhorn
    W_sfs2, mask_sfs2 = sfs_sinq_sparse_v2(W, X, sparsity, nbits, group_size)
    mse_sfs2 = ((W - W_sfs2) ** 2).mean().item()
    improvement2 = (mse_baseline - mse_sfs2) / mse_baseline * 100
    print(f"SFS v2 MSE:   {mse_sfs2:.6f} ({improvement2:+.2f}%)")

    return mse_baseline, mse_sfs1, mse_sfs2


def test_sfs_across_sparsity():
    """Test SFS across different sparsity levels."""
    print("\n" + "="*60)
    print("TEST: SFS vs Sparsity Level")
    print("="*60)

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    K, N = 128, 256
    batch = 32
    nbits = 3
    group_size = 64

    W = torch.randn(K, N, device=device)
    X = torch.randn(batch, N, device=device)

    print(f"\n{'Sparsity':>10s} | {'Baseline':>12s} | {'SFS v1':>12s} | {'SFS v2':>12s} | {'Best Impr':>10s}")
    print("-" * 65)

    for sparsity in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5]:
        W_baseline, _ = baseline_sinq_sparse(W.clone(), X, sparsity, nbits, group_size)
        mse_baseline = ((W - W_baseline) ** 2).mean().item()

        W_sfs1, _ = sfs_sinq_sparse(W.clone(), X, sparsity, nbits, group_size)
        mse_sfs1 = ((W - W_sfs1) ** 2).mean().item()

        W_sfs2, _ = sfs_sinq_sparse_v2(W.clone(), X, sparsity, nbits, group_size)
        mse_sfs2 = ((W - W_sfs2) ** 2).mean().item()

        best_mse = min(mse_sfs1, mse_sfs2)
        best_impr = (mse_baseline - best_mse) / mse_baseline * 100

        print(f"{sparsity*100:>9.0f}% | {mse_baseline:>12.6f} | {mse_sfs1:>12.6f} | {mse_sfs2:>12.6f} | {best_impr:>+9.2f}%")


def test_sfs_on_real_weights():
    """Test SFS on real Qwen-0.5B weights."""
    print("\n" + "="*60)
    print("TEST: SFS on Real Qwen-0.5B Weights")
    print("="*60)

    try:
        from transformers import AutoModelForCausalLM

        print("Loading Qwen-0.5B...")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Test on layer 0 gate_proj
        W = model.model.layers[0].mlp.gate_proj.weight.data.float()
        K, N = W.shape
        device = W.device

        print(f"\nTesting on layer 0 gate_proj [{K}x{N}]")

        batch = 64
        X = torch.randn(batch, N, device=device, dtype=torch.float32)

        sparsity = 0.35
        nbits = 3
        group_size = 64

        # Baseline
        W_baseline, _ = baseline_sinq_sparse(W, X, sparsity, nbits, group_size)
        mse_baseline = ((W - W_baseline) ** 2).mean().item()

        # SFS v1
        W_sfs1, _ = sfs_sinq_sparse(W, X, sparsity, nbits, group_size)
        mse_sfs1 = ((W - W_sfs1) ** 2).mean().item()

        # SFS v2
        W_sfs2, _ = sfs_sinq_sparse_v2(W, X, sparsity, nbits, group_size)
        mse_sfs2 = ((W - W_sfs2) ** 2).mean().item()

        print(f"\nResults at {sparsity*100:.0f}% sparsity, {nbits}-bit:")
        print(f"Baseline MSE: {mse_baseline:.6f}")
        print(f"SFS v1 MSE:   {mse_sfs1:.6f} ({(mse_baseline - mse_sfs1) / mse_baseline * 100:+.2f}%)")
        print(f"SFS v2 MSE:   {mse_sfs2:.6f} ({(mse_baseline - mse_sfs2) / mse_baseline * 100:+.2f}%)")

        del model
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    test_sfs()
    test_sfs_across_sparsity()
    test_sfs_on_real_weights()

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print("""
If SFS shows improvement:
- The insight is that Sinkhorn factors should be computed for sparse matrix
- This is a simple but potentially impactful change

If SFS shows NO improvement:
- Sinkhorn on sparse matrix may produce suboptimal factors
- Need different approach
    """)
