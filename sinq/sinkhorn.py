'''
written by SINQ authors
'''
import torch


def sinkhorn_log(matrix,
                 order=8,
                 clip_min=1e-3,
                 clip_max=1e3,
                 eps=1e-6,
                 stop_on_increasing_imbalance=True):
    """
    vmap-friendly Sinkhorn that returns *the* mu1 / mu2 corresponding
    to the matrix with the minimal imbalance encountered during the
    iteration.

    The return value is a tuple
        (scaled_matrix, mu1_at_minimum, mu2_at_minimum)
    """
    dtype = torch.float32
    m = matrix.to(dtype)
    dev = m.device
    measure = torch.std

    def imbalance(mat):
        s1, s2 = measure(mat, 1), measure(mat, 0)
        s_min = torch.minimum(s1.min(), s2.min()).clamp_min(1e-12)
        s_max = torch.maximum(s1.max(), s2.max())
        return s_max / s_min          # scalar

    imb_min = torch.tensor(float('inf'), dtype=dtype, device=dev)
    gate    = torch.tensor(0.0, dtype=dtype, device=dev)

    tgt_small = torch.minimum(
        m.std(1).clamp(clip_min, clip_max).min(),
        m.std(0).clamp(clip_min, clip_max).min()
    ) + eps

    log_mu1 = torch.zeros(m.shape[1], dtype=dtype, device=dev)
    log_mu2 = torch.zeros(m.shape[0], 1, dtype=dtype, device=dev)

    # Known-good candidates for the step k=0
    cur0          = m
    ib0           = imbalance(cur0)
    imb_min       = torch.minimum(imb_min, ib0)
    mu1_star      = log_mu1.exp().clone()
    mu2_star      = log_mu2.exp().clone()

    for _ in range(order):
        cur       = (m / log_mu1.exp()) / log_mu2.exp()
        ib        = imbalance(cur)

        # update the best-so-far candidates
        better    = (ib <= imb_min).to(dtype)   # 1 if new best
        imb_min   = torch.min(imb_min, ib)
        mu1_star  = torch.where(better.bool(), log_mu1.exp(), mu1_star)
        mu2_star  = torch.where(better.bool(), log_mu2.exp(), mu2_star)

        # early-exit condition
        if stop_on_increasing_imbalance:
            rising = (ib > imb_min).to(dtype)
            gate   = torch.clip(gate + rising, max=1.0)   # once 1 → always 1

        # still-running samples update the dual variables
        g  = 1.0 - gate

        std_r  = measure(cur, 1).clamp(clip_min, clip_max)
        std_c  = measure(cur,0).clamp(clip_min, clip_max)

        sal_col = (std_c / tgt_small).clamp(0.7, 2.0).log()
        sal_row = (std_r[:, None] / tgt_small).clamp(0.7, 2.0).log()

        log_mu1 = (log_mu1 + (sal_col * g)).clip(-.3, 10.)
        log_mu2 = (log_mu2 + (sal_row * g)).clip(-.3, 10.)

    # final scaled matrix and the recorded best scaling vectors
    scaled = m / mu1_star / mu2_star
    return scaled, mu1_star, mu2_star


def kurtosis(x, fisher=True, dim=1, clip_var=False, keepdim=False):
    """
    Calculate the kurtosis of a tensor along specified dimensions.

    Args:
        x: Input tensor
        fisher: If True (default), returns Fisher's definition (kurtosis - 3)
        dim: Dimension or dimensions to reduce. If None, reduces all dimensions.
        keepdim: Whether the output tensor has dim retained or not.

    Returns:
        Kurtosis values
    """
    if dim is None:
        # Reduce all dimensions
        dim = tuple(range(x.dim()))

    mean = torch.mean(x, dim=dim, keepdim=True)
    diffs = x - mean
    var = torch.mean(diffs ** 2, dim=dim, keepdim=True)
    if clip_var:
        var = torch.clamp(var, min=1e-8)
    std = torch.sqrt(var)
    zscores = diffs / std
    kurt = torch.mean(zscores ** 4, dim=dim, keepdim=keepdim)

    if fisher:
        kurt -= 3

    return kurt

def min_kurt_vectors_vmap(M, n_iter=50, dim=0, lr=1e-2, momentum=0.7):
    """
    Vectorized version of min_kurt_vectors for use with vmap.
    Avoids autograd and in-place operations for vmap compatibility.
    """
    # note: only tested with 2D tiling
    K, L = M.shape
    device = M.device
    dtype = M.dtype

    mu1 = (torch.ones(1, L, device=device)).double()
    mu2 = (torch.ones(K, 1, device=device)).double()

    # Initialize momentum terms
    momentum_mu1 = torch.zeros_like(mu1)
    momentum_mu2 = torch.zeros_like(mu2)
    M = M.detach().double()  # Detach M from any computation graph

    # Helper function for explicit kurtosis gradient
    def kurtosis_grad_col(x, dim=1):
        K_val = x.shape[0]
        m = torch.mean(x, dim=dim, keepdim=True)
        v = torch.var(x, correction=0, dim=dim, keepdim=True)  # Divide by K (not K-1)
        v = torch.clamp(v, min=1e-8)
        std = torch.sqrt(v)
        z = (x - m) / std
        term1 = z**3
        sum_term1 = torch.sum(term1, dim=dim, keepdim=True)
        term2 = z**4
        sum_term2 = torch.sum(term2, dim=dim, keepdim=True)
        grad = (4/(K_val * std)) * (term1 - (1/K_val)*sum_term1 - (1/K_val)*z * sum_term2)
        return grad

    # Optimization loop
    for _ in range(n_iter):
        # Forward pass
        y =  M / mu1 / mu2  # (K, L)

        grad_y = kurtosis_grad_col(y, 0)/K + kurtosis_grad_col(y,1)/L

        # Compute gradients for mu1 and mu2
        grad_mu1 = -(grad_y * mu2 * M).sum(dim=0, keepdim=True)  # (1, L)
        grad_mu2 = -(grad_y * M * mu1).sum(dim=1, keepdim=True)  # (K, 1)

        # Update momentum terms
        momentum_mu1 = momentum * momentum_mu1 + grad_mu1
        momentum_mu2 = momentum * momentum_mu2 + grad_mu2

        # Update parameters
        mu1 = (mu1 - lr * momentum_mu1 ).clip(0.1, 1e4)
        mu2 = (mu2 - lr * momentum_mu2 ).clip(0.1, 1e4)

    return (M / mu1 / mu2).to(dtype), (mu1).to(dtype), (mu2).to(dtype)



