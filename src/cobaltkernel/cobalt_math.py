"""Canonical CoBALT math, vectorised, for the memory-streamed layer-wise quantizer.

Reproduces exactly (verified against the cited lines):
  * Wanda importance  |W| * ||X_j||_2                      src/nosink.py:283-287
  * row+col quantile self-normalisation + ONE global top-k  src/nosink.py:130 / :77
  * OBS compensation with H = X^T X + lambda I              sinq/sparse_quant.py:133
  * sparse-aware per-column scale + robust clamp            src/nosink.py:619/633/644
  * group RTN (asymmetric unsigned min-max, g=128)          sinq/dual_shift.py:132-155
                                                            src/eout_quant.py:71 (survivor hull)
"""
import torch


# ---------------------------------------------------------------- mask
def balanced_keepmask(imp, sparsity, col_exp):
    """CoBALT balanced keep-mask (row quantile / col^beta quantile / global top-k).
    Byte-identical to nosink.balanced_mask_and_obs's mask branch (row_fair=True,
    per_row_thresh=False, scope='global')."""
    K, N = imp.shape
    if sparsity <= 0.0:
        return torch.ones_like(imp)
    kr, kc = int(N * sparsity), int(K * sparsity)
    if kr > 0:
        imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
        imp = _rescale(imp)      # global positive rescale: mask-invariant, prevents fp32 overflow
    if col_exp > 0 and kc > 0:
        imp = imp / torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30).pow(col_exp)
        imp = _rescale(imp)
    n_prune = int(K * N * sparsity)
    flat = imp.reshape(-1)
    thr = _kth_global(flat, n_prune)
    return (flat > thr).view(K, N).float()


def blocked_keepmask(imp, sparsity, col_exp, block=32):
    """CoBALT-16:32 fixed-cardinality block keep-mask.

    Normalisation pipeline is BYTE-IDENTICAL to `balanced_keepmask` (row kthvalue /
    `_rescale` / col^beta kthvalue / `_rescale`); only the final selection changes:
    instead of ONE global top-k over all K*N entries, each aligned block of `block`
    input columns keeps exactly `keep = block - int(block*sparsity)` entries
    (block=32, sparsity=0.5 -> exactly 16 of 32). Overall sparsity is therefore
    exactly int(block*sparsity)/block in every row of every matrix.

    Ties are broken by torch.topk's default ordering (deterministic)."""
    K, N = imp.shape
    if sparsity <= 0.0:
        return torch.ones_like(imp)
    assert N % block == 0, f"N={N} not divisible by mask_block={block}"
    kr, kc = int(N * sparsity), int(K * sparsity)
    if kr > 0:
        imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
        imp = _rescale(imp)
    if col_exp > 0 and kc > 0:
        imp = imp / torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30).pow(col_exp)
        imp = _rescale(imp)
    keep = block - int(block * sparsity)
    nb = N // block
    v = imp.reshape(K, nb, block)
    idx = torch.topk(v, keep, dim=-1).indices
    mask = torch.zeros_like(v)
    mask.scatter_(-1, idx, 1.0)
    return mask.reshape(K, N)


def _rescale(imp):
    """Divide by the global max. A single positive scalar does NOT change the global
    top-k support (nor the subsequent column quantile ratios), but it keeps the row/col
    quantile self-normalisation from overflowing fp32 to +inf -- which happens on
    MedGemma-27B whenever a row AND a column quantile are both near the 1e-30 clamp
    (dead input channels / near-zero rows). An inf importance makes `imp > thr` False
    everywhere => an ALL-ZERO mask (measured on 27B layers 25/30/33)."""
    m = imp.amax()
    if torch.isfinite(m) and m > 0:
        return imp / m
    return torch.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)


def _kth_global(flat, k):
    """kthvalue over a possibly huge flat tensor. torch.kthvalue is limited to
    2**31 elements and is slow on CUDA for >1e8; use a sort-free selection via
    torch.topk on a coarse histogram refinement when the tensor is large."""
    n = flat.numel()
    if n <= 32_000_000:
        return torch.kthvalue(flat, k).values
    # two-pass histogram selection (exact): bracket the k-th smallest value.
    lo = flat.min().float()
    hi = flat.max().float()
    # values are non-negative importances; refine the bracket ~40 times => exact to fp32 ulp
    for _ in range(64):
        mid = (lo + hi) * 0.5
        if not torch.isfinite(mid) or mid <= lo or mid >= hi:
            break
        cnt = (flat <= mid).sum()
        if cnt.item() < k:
            lo = mid
        else:
            hi = mid
    # exact refinement: the k-th smallest is the largest value <= hi
    sel = flat[flat <= hi]
    return sel.max() if sel.numel() else hi


# ---------------------------------------------------------------- OBS
def obs_compensate(W, mask, H, damping_frac=0.01, damping_min=1e-2):
    """W_comp = (W - ((W*(1-mask)) / diag(Hinv)) @ Hinv) * mask.
    Identical math to nosink.py:293-296's per-row python loop (H symmetric)."""
    N = H.shape[0]
    damping = max(damping_frac * float(H.diagonal().mean()), damping_min)
    Hd = H.clone()
    Hd.diagonal().add_(damping)
    try:
        L = torch.linalg.cholesky(Hd)
        Hinv = torch.cholesky_inverse(L)
        method = "cholesky"
    except Exception:
        Hinv = torch.linalg.inv(Hd)
        method = "inv"
    del Hd
    d = Hinv.diagonal().clamp(min=1e-12)
    P = W * (1.0 - mask)
    W_comp = (W - (P / d.view(1, -1)) @ Hinv) * mask
    del P, Hinv
    return W_comp, damping, method


# ---------------------------------------------------------------- col scale
def masked_std(W, mask, dim, eps=1e-8):
    cnt = mask.sum(dim=dim).clamp(min=1.0)
    Wm = W * mask
    mean = Wm.sum(dim=dim) / cnt
    centered = (W - (mean.view(1, -1) if dim == 0 else mean.view(-1, 1))) * mask
    var = (centered ** 2).sum(dim=dim) / cnt
    return var.sqrt().clamp(min=eps)


def robust_scale(s, ratio=10.0):
    s = torch.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-8)
    gm = s.log().mean().exp()
    return (s / gm).clamp(min=1.0 / ratio, max=ratio)


# ---------------------------------------------------------------- group RTN
def group_rtn(W_norm, mask, nbits, gsize, hull="survivor"):
    """Asymmetric unsigned min-max group RTN along dim=1.
      hull='all'      : min/max over ALL entries of the group (pruned = exact 0)   [nosink]
      hull='survivor' : min/max over SURVIVORS only                                [eout repack]
    Returns q[K,N] float, scale[K,ng], zero[K,ng]."""
    K, N = W_norm.shape
    assert N % gsize == 0, f"N={N} not divisible by gsize={gsize}"
    ng = N // gsize
    Wg = W_norm.view(K, ng, gsize)
    n_levels = 2 ** nbits - 1
    if hull == "survivor":
        Mg = mask.view(K, ng, gsize).bool()
        big = torch.finfo(torch.float32).max
        w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
        w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
        empty = ~Mg.any(-1, keepdim=True)
        w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
        w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    else:
        w_min = Wg.amin(-1, keepdim=True)
        w_max = Wg.amax(-1, keepdim=True)
    scale = (w_max - w_min).clamp(min=1e-4) / n_levels
    zero = -torch.round(w_min / scale)
    q = torch.clamp(torch.round(Wg / scale + zero), 0, n_levels)
    return q.view(K, N), scale.view(K, ng), zero.view(K, ng)


def dequant(q, scale, zero, col_scale, mask, gsize):
    K, N = q.shape
    ng = N // gsize
    W = (q.view(K, ng, gsize).float() - zero.view(K, ng, 1).float()) * scale.view(K, ng, 1).float()
    return W.view(K, N) * col_scale.view(1, -1) * mask


# ---------------------------------------------------------------- full recipe
def cobalt_quantize(W, H, sparsity, beta, nbits, gsize, hull="survivor",
                    damping_frac=0.01, damping_min=1e-2, mask_block=0):
    """One nn.Linear weight [K=out, N=in]. H is the RAW (undamped) fp32 Gram X^T X
    accumulated over ALL calibration tokens; act_norms = sqrt(diag H)."""
    K, N = W.shape
    W = W.float()
    act_norms = H.diagonal().clamp(min=0).sqrt()                     # ||X_j||_2
    imp = W.abs() * act_norms.view(1, -1)                            # Wanda
    if mask_block and mask_block > 0:
        mask = blocked_keepmask(imp, sparsity, beta, mask_block)
    else:
        mask = balanced_keepmask(imp, sparsity, beta)
    del imp
    W_comp, damping, inv_method = obs_compensate(W, mask, H, damping_frac, damping_min)
    c = robust_scale(masked_std(W_comp, mask, dim=0))                # norm='col'
    W_norm = W_comp / c.view(1, -1)
    del W_comp
    if float(mask.sum()) == 0.0 and sparsity < 1.0:
        raise RuntimeError("degenerate all-zero keep-mask (importance overflow?)")
    q, scale, zero = group_rtn(W_norm, mask, nbits, gsize, hull)
    del W_norm
    W_hat = dequant(q, scale, zero, c, mask, gsize)
    D = W_hat - W
    relerr = float(D.norm() / W.norm().clamp(min=1e-12))
    # output error tr(D H D^T) / tr(W H W^T), fp32 (H can be ill-conditioned; use fp64 reduce)
    hs = H.diagonal().mean().clamp(min=1e-30)
    Hn = H / hs
    num = float(((D @ Hn) * D).double().sum())
    den = float(((W @ Hn) * W).double().sum())
    del Hn
    stats = dict(sparsity_achieved=float(1.0 - mask.mean()), relerr=relerr,
                 eout_ratio=(num / den if den > 0 else float("nan")),
                 damping=damping, inv_method=inv_method,
                 dead_cols=int((mask.sum(0) == 0).sum()), dead_rows=int((mask.sum(1) == 0).sum()))
    return q, scale, zero, c, mask, W_hat, stats
