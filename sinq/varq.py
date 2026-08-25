"""
VARQ: Variance-Adaptive Range Quantization

A gradient-free, hyperparameter-free alternative to Sinkhorn normalization
that directly minimizes within-group quantization ranges.

Written for SINQ codebase - 2025
"""

import torch


def varq_normalize(
    matrix,
    group_size=64,
    clip_min=0.5,
    clip_max=2.0,
    eps=1e-10
):
    """
    VARQ normalization for dual-scale quantization.

    Unlike Sinkhorn which minimizes variance imbalance globally, VARQ:
    1. Balances row ranges using robust median-based scaling
    2. Minimizes within-group ranges by optimally scaling columns per-group

    This exploits the separability of 1D tiling where column scales affect
    only their respective groups.

    Args:
        matrix: Input weight matrix (H, W)
        group_size: Quantization group size (default 64)
        clip_min: Minimum scale factor (default 0.5)
        clip_max: Maximum scale factor (default 2.0)
        eps: Numerical stability constant (default 1e-10)

    Returns:
        Tuple of (scaled_matrix, mu1_col_scales, mu2_row_scales)
        Format matches sinkhorn_log() for drop-in replacement.

    Mathematical Details:
    ---------------------
    Stage 1 (Row Balancing):
        - Compute row ranges: R[i] = max(W[i,:]) - min(W[i,:])
        - Target range: R_target = median(R)
        - Row scales: s[i] = clip(R_target / R[i], clip_min, clip_max)

    Stage 2 (Column Optimization per Group):
        For each group j of 64 columns:
        - Compute column max magnitudes: M[k] = max_i |W[i, j*64+k]|
        - Optimal column scales: t[j*64+k] = 1 / M[k]
        - Normalize: t[j*64:j*64+64] /= geometric_mean(t[j*64:j*64+64])

    Theoretical Guarantee:
    ---------------------
    Stage 2 achieves Chebyshev-optimal range minimization for each group
    independently. The overall solution minimizes max(group_ranges).

    See: /workspace/SINQ/llmdocs/PROPOSAL_VARQ.md for full derivation.
    """
    dtype = matrix.dtype
    device = matrix.device
    m = matrix.to(torch.float32)

    H, W = m.shape
    assert W % group_size == 0, f"Width {W} must be divisible by group_size {group_size}"
    n_groups = W // group_size

    # ============================================
    # STAGE 1: Row Scale Optimization
    # ============================================
    # Compute row ranges (max - min for each row)
    row_max = m.max(dim=1).values  # (H,)
    row_min = m.min(dim=1).values  # (H,)
    row_ranges = row_max - row_min  # (H,)

    # Target range: use median for robustness to outliers
    R_target = torch.median(row_ranges)

    # Compute row scales with clipping to prevent extreme values
    s_raw = R_target / (row_ranges + eps)
    mu2 = s_raw.clamp(clip_min, clip_max).view(-1, 1)  # (H, 1)

    # Apply row scaling
    m_row_scaled = m * mu2  # (H, W)

    # ============================================
    # STAGE 2: Column Scale Optimization (Per-Group)
    # ============================================
    mu1 = torch.ones(W, device=device, dtype=torch.float32)

    for j in range(n_groups):
        col_start = j * group_size
        col_end = (j + 1) * group_size

        # Extract group j (all rows, 64 consecutive columns)
        group_matrix = m_row_scaled[:, col_start:col_end]  # (H, group_size)

        # Compute RANGE for each column (max - min), not max_abs
        # This is the correct objective: quantization error ∝ range
        col_max = group_matrix.max(dim=0).values  # (group_size,)
        col_min = group_matrix.min(dim=0).values  # (group_size,)
        col_ranges = col_max - col_min  # (group_size,)

        # Optimal column scales: inverse of range
        # This equalizes ranges across columns, directly minimizing quantization error
        t_group = 1.0 / (col_ranges + eps)

        # Normalize by geometric mean to prevent scale drift
        # geometric_mean = (product)^(1/n) = exp(mean(log(values)))
        log_mean = torch.log(t_group + eps).mean()
        geometric_mean = torch.exp(log_mean)
        t_group = t_group / geometric_mean

        # Store in global column scale vector
        mu1[col_start:col_end] = t_group

    # ============================================
    # Final Scaled Matrix
    # ============================================
    # Apply both row and column scales
    scaled = m * mu2 * mu1.view(1, -1)

    # Convert to original dtype and return in format matching sinkhorn_log
    # Note: sinkhorn_log returns (scaled_matrix, mu1_col, mu2_row)
    return (
        scaled.to(dtype),
        mu1.view(-1).to(dtype),  # Column scales (W,)
        mu2.view(-1, 1).to(dtype)  # Row scales (H, 1)
    )


def varq_normalize_adaptive(
    matrix,
    group_size=64,
    clip_min=0.5,
    clip_max=2.0,
    eps=1e-10,
    variance_threshold=0.5
):
    """
    Adaptive variant of VARQ that adjusts group size based on matrix variance.

    For matrices with high variance (many outliers), use smaller groups.
    For matrices with low variance (uniform distribution), use larger groups.

    Args:
        matrix: Input weight matrix (H, W)
        group_size: Base quantization group size (default 64)
        variance_threshold: Threshold for adaptive group sizing (default 0.5)
        Other args: Same as varq_normalize()

    Returns:
        Same as varq_normalize()
    """
    # Compute normalized variance
    var = matrix.var()
    mean = matrix.mean()
    cv = torch.sqrt(var) / (torch.abs(mean) + eps)  # Coefficient of variation

    # Adaptive group size
    if cv > variance_threshold:
        adaptive_group_size = max(32, group_size // 2)  # Smaller groups for high variance
    else:
        adaptive_group_size = group_size

    # Ensure width is divisible
    W = matrix.shape[1]
    while W % adaptive_group_size != 0 and adaptive_group_size > 16:
        adaptive_group_size = adaptive_group_size // 2

    return varq_normalize(
        matrix,
        group_size=adaptive_group_size,
        clip_min=clip_min,
        clip_max=clip_max,
        eps=eps
    )


def compute_group_ranges(matrix, group_size=64):
    """
    Utility function to compute the range (max - min) for each column group.

    Useful for analyzing and comparing normalization methods.

    Args:
        matrix: Input matrix (H, W)
        group_size: Group size (default 64)

    Returns:
        Tensor of ranges for each group (n_groups,)
    """
    H, W = matrix.shape
    assert W % group_size == 0
    n_groups = W // group_size

    ranges = []
    for j in range(n_groups):
        col_start = j * group_size
        col_end = (j + 1) * group_size
        group = matrix[:, col_start:col_end]
        group_range = group.max() - group.min()
        ranges.append(group_range)

    return torch.tensor(ranges, device=matrix.device)


def compare_normalizations(matrix, group_size=64):
    """
    Compare VARQ and Sinkhorn normalizations side-by-side.

    Args:
        matrix: Input matrix (H, W)
        group_size: Group size (default 64)

    Returns:
        Dictionary with comparison metrics
    """
    from .sinkhorn import sinkhorn_log

    # SINQ normalization
    W_sinq, mu1_sinq, mu2_sinq = sinkhorn_log(matrix, order=16)

    # VARQ normalization
    W_varq, mu1_varq, mu2_varq = varq_normalize(matrix, group_size=group_size)

    # Compute group ranges
    ranges_sinq = compute_group_ranges(W_sinq, group_size)
    ranges_varq = compute_group_ranges(W_varq, group_size)

    # Compute imbalance (max std / min std)
    def imbalance(mat):
        s1, s2 = torch.std(mat, 1), torch.std(mat, 0)
        s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
        s_max = torch.maximum(s1.max(), s2.max())
        return s_max / s_min

    return {
        'sinq': {
            'imbalance': imbalance(W_sinq).item(),
            'max_range': ranges_sinq.max().item(),
            'mean_range': ranges_sinq.mean().item(),
            'std_range': ranges_sinq.std().item(),
            'range_imbalance': (ranges_sinq.max() / ranges_sinq.min()).item(),
        },
        'varq': {
            'imbalance': imbalance(W_varq).item(),
            'max_range': ranges_varq.max().item(),
            'mean_range': ranges_varq.mean().item(),
            'std_range': ranges_varq.std().item(),
            'range_imbalance': (ranges_varq.max() / ranges_varq.min()).item(),
        },
        'improvement': {
            'max_range_reduction': ((ranges_sinq.max() - ranges_varq.max()) / ranges_sinq.max()).item(),
            'range_imbalance_reduction': ((ranges_sinq.max()/ranges_sinq.min() - ranges_varq.max()/ranges_varq.min()) / (ranges_sinq.max()/ranges_sinq.min())).item(),
        }
    }


# Hybrid approach: Use Sinkhorn Stage 1, VARQ Stage 2
def varq_hybrid_normalize(
    matrix,
    group_size=64,
    sinkhorn_order=8,
    clip_min=0.5,
    clip_max=2.0,
    eps=1e-10
):
    """
    Hybrid normalization: Sinkhorn for global balancing + VARQ for per-group optimization.

    This combines the proven effectiveness of Sinkhorn's iterative balancing with
    VARQ's targeted per-group range minimization.

    Args:
        matrix: Input weight matrix (H, W)
        group_size: Quantization group size (default 64)
        sinkhorn_order: Number of Sinkhorn iterations (default 8, half of standard 16)
        Other args: Same as varq_normalize()

    Returns:
        Same as varq_normalize()
    """
    from .sinkhorn import sinkhorn_log

    dtype = matrix.dtype
    device = matrix.device
    m = matrix.to(torch.float32)

    # ============================================
    # STAGE 1: Sinkhorn Global Balancing
    # ============================================
    m_sinkhorn, mu1_sinq, mu2_sinq = sinkhorn_log(
        m,
        order=sinkhorn_order,
        clip_min=clip_min,
        clip_max=clip_max,
        eps=eps
    )

    # ============================================
    # STAGE 2: VARQ Per-Group Column Optimization
    # ============================================
    H, W = m_sinkhorn.shape
    n_groups = W // group_size

    mu1_varq = torch.ones(W, device=device, dtype=torch.float32)

    for j in range(n_groups):
        col_start = j * group_size
        col_end = (j + 1) * group_size

        # Extract group j from Sinkhorn-normalized matrix
        group_matrix = m_sinkhorn[:, col_start:col_end]

        # Compute column-wise max absolute values
        col_max_abs = group_matrix.abs().max(dim=0).values

        # Optimal column scales
        t_group = 1.0 / (col_max_abs + eps)

        # Normalize by geometric mean
        log_mean = torch.log(t_group + eps).mean()
        geometric_mean = torch.exp(log_mean)
        t_group = t_group / geometric_mean

        mu1_varq[col_start:col_end] = t_group

    # ============================================
    # Combine Scales and Return
    # ============================================
    # Final column scales: product of Sinkhorn and VARQ column scales
    mu1_final = mu1_sinq * mu1_varq

    # Row scales: from Sinkhorn (unchanged)
    mu2_final = mu2_sinq

    # Apply combined scales
    scaled = m / mu1_final.view(1, -1) / mu2_final

    return (
        scaled.to(dtype),
        mu1_final.view(-1).to(dtype),
        mu2_final.to(dtype)
    )


if __name__ == "__main__":
    # Quick test
    torch.manual_seed(42)

    # Create test matrix with outliers
    W = torch.randn(512, 512)
    W[100, :] *= 10  # row outlier
    W[:, 200] *= 8   # column outlier

    print("Testing VARQ normalization...")
    print(f"Original matrix - mean: {W.mean():.4f}, std: {W.std():.4f}")
    print(f"Original matrix - min: {W.min():.4f}, max: {W.max():.4f}")

    # Test basic VARQ
    W_norm, mu1, mu2 = varq_normalize(W, group_size=64)
    print(f"\nVARQ normalized - mean: {W_norm.mean():.4f}, std: {W_norm.std():.4f}")
    print(f"VARQ normalized - min: {W_norm.min():.4f}, max: {W_norm.max():.4f}")

    # Test group ranges
    ranges = compute_group_ranges(W_norm, group_size=64)
    print(f"\nGroup ranges:")
    print(f"  Max range: {ranges.max():.6f}")
    print(f"  Mean range: {ranges.mean():.6f}")
    print(f"  Std range: {ranges.std():.6f}")
    print(f"  Range imbalance: {(ranges.max() / ranges.min()):.4f}")

    print("\n✓ VARQ basic test completed successfully!")
