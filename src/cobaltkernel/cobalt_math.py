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


# ---------------------------------------------------------------- scale/zero search (quantizer-stage lever)
# Fixed GLOBAL candidate grid: rho shrinks the coded half-range, delta shifts its centre (in units of the
# min-max half-range). Per-group argmin of a weighted squared error over SURVIVORS. Stores exactly the same
# (scale, zero) per group as min-max RTN => bpw-identical, same decoder.
CLIP_RHOS = (1.0, 0.975, 0.95, 0.925, 0.90, 0.875, 0.85, 0.825, 0.80)
CLIP_DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)


def find_params(Wg, Mg, n_levels, clip="none", wcol=None, hull="survivor"):
    """Per-group (scale, zero) for Wg [K, G, g] with survivor mask Mg [K, G, g] (bool).
      clip='none' : min-max over the hull (== group_rtn, bit-identical)
      clip='mse'  : grid argmin of  sum_j (W - W_hat)^2            over survivors
      clip='aw'   : grid argmin of  sum_j wcol_j (W - W_hat)^2     (activation-weighted; wcol [G, g] = ||X_j||^2
                    in the quantized space = output-error diagonal)
    Returns scale [K,G,1], zero [K,G,1]."""
    big = torch.finfo(torch.float32).max
    if hull == "survivor":
        w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
        w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
        empty = ~Mg.any(-1, keepdim=True)
        w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
        w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    else:
        w_min = Wg.amin(-1, keepdim=True)
        w_max = Wg.amax(-1, keepdim=True)
    if clip == "none":
        scale = (w_max - w_min).clamp(min=1e-4) / n_levels
        zero = -torch.round(w_min / scale)
        return scale, zero
    mid0 = 0.5 * (w_min + w_max)
    half0 = (0.5 * (w_max - w_min)).clamp(min=5e-5)
    wt = Mg.float() if clip == "mse" else (Mg.float() * wcol.unsqueeze(0))
    best_e = best_s = best_z = None
    for rho in CLIP_RHOS:
        for dl in CLIP_DELTAS:
            half = half0 * rho
            lo = (mid0 + dl * half0) - half
            scale = (2.0 * half) / n_levels
            zero = -torch.round(lo / scale)
            q = torch.clamp(torch.round(Wg / scale + zero), 0, n_levels)
            e = (((q - zero) * scale - Wg) ** 2 * wt).sum(-1, keepdim=True)
            if best_e is None:
                best_e, best_s, best_z = e, scale, zero
            else:
                take = e < best_e
                best_e = torch.where(take, e, best_e)
                best_s = torch.where(take, scale, best_s)
                best_z = torch.where(take, zero, best_z)
    return best_s, best_z


def group_quant(W_norm, mask, nbits, gsize, hull="survivor", clip="none", wcol=None):
    """group_rtn generalised with the clip/zero search; clip='none' is bit-identical to group_rtn."""
    K, N = W_norm.shape
    ng = N // gsize
    Wg = W_norm.view(K, ng, gsize)
    Mg = mask.view(K, ng, gsize).bool()
    n_levels = 2 ** nbits - 1
    wc = wcol.view(ng, gsize) if wcol is not None else None
    scale, zero = find_params(Wg, Mg, n_levels, clip, wc, hull)
    q = torch.clamp(torch.round(Wg / scale + zero), 0, n_levels)
    return q.view(K, N), scale.view(K, ng), zero.view(K, ng)


# ---------------------------------------------------------------- GPTQ-style sequential rounding compensation
def gptq_quantize(W_norm, mask, Hn, nbits, gsize, hull="survivor", damping_frac=0.01, damping_min=1e-2,
                  clip="none", actorder=False, blocksize=128):
    """Sequential column-wise OBS error feedback (GPTQ / SparseGPT loop) with a FIXED keep-mask.

    Processes columns in order; a pruned entry is set to 0, a survivor is rounded to its group's grid;
    the error (w - w_hat)/[H^-1]_jj is propagated to every not-yet-processed column through the Cholesky
    factor of H^-1 (exact inverse-Hessian of the remaining columns). For the PRUNING part this reproduces
    the joint OBS solution of `obs_compensate` exactly (sequential elimination of a quadratic); on top it
    compensates the ROUNDING error of survivors, which plain RTN leaves uncompensated.
    W_norm is in the column-scaled space; Hn must be the matching Gram diag(c) H diag(c).
      actorder=False : groups are dynamic (scale/zero from the already-updated weights at group start)
      actorder=True  : columns processed by descending diag(H); groups STATIC (params from the OBS-compensated,
                       unquantized survivors, per original group) so the stored contiguous-group format holds.
    Returns q[K,N] (0 at pruned), scale[K,ng], zero[K,ng], W_hat_norm[K,N] (sequentially compensated)."""
    K, N = W_norm.shape
    ng = N // gsize
    assert blocksize % gsize == 0 or gsize % blocksize == 0
    n_levels = 2 ** nbits - 1
    W = W_norm.clone().float()
    M = mask.bool()
    H = Hn.clone().float()
    dead = H.diagonal() == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damping = max(damping_frac * float(H.diagonal().mean()), damping_min)
    H.diagonal().add_(damping)
    wcol_full = H.diagonal().clone()
    perm = None
    static = None
    if actorder:
        perm = torch.argsort(H.diagonal(), descending=True)
        # static params from the exact joint-OBS compensated survivors (what RTN would quantize)
        W_comp, _, _ = obs_compensate(W, mask, Hn, damping_frac, damping_min)
        s0, z0 = find_params(W_comp.view(K, ng, gsize), M.view(K, ng, gsize), n_levels, clip,
                             wcol_full.view(ng, gsize), hull)
        static = (s0.view(K, ng), z0.view(K, ng))
        del W_comp
        W = W[:, perm]; M = M[:, perm]; H = H[perm][:, perm]
        invperm = torch.argsort(perm)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    del L
    Q = torch.zeros_like(W)          # dequantized (normalized space, permuted order)
    C = torch.zeros_like(W)          # integer codes
    scale = torch.zeros(K, ng, device=W.device)
    zero = torch.zeros(K, ng, device=W.device)
    cur_s = cur_z = None
    for i1 in range(0, N, blocksize):
        i2 = min(i1 + blocksize, N)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        M1 = M[:, i1:i2]
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            if actorder:
                g = (perm[col] // gsize)
                s = static[0][:, g].unsqueeze(1); z = static[1][:, g].unsqueeze(1)
            else:
                if col % gsize == 0:
                    g = col // gsize
                    # group params from the CURRENT (updated) weights of the next gsize columns
                    if i + gsize <= cnt:
                        Wgc = W1[:, i:i + gsize]
                        Mgc = M1[:, i:i + gsize]
                    else:
                        Wgc = torch.cat([W1[:, i:], W[:, i2:col + gsize]], 1)
                        Mgc = M[:, col:col + gsize]
                    s, z = find_params(Wgc.unsqueeze(1), Mgc.unsqueeze(1), n_levels, clip,
                                       wcol_full[col:col + gsize].view(1, gsize), hull)
                    cur_s, cur_z = s.view(K, 1), z.view(K, 1)
                    scale[:, g] = cur_s[:, 0]; zero[:, g] = cur_z[:, 0]
                s, z = cur_s, cur_z
            w = W1[:, i].unsqueeze(1)
            m = M1[:, i].unsqueeze(1)
            qc = torch.clamp(torch.round(w / s + z), 0, n_levels)
            wq = (qc - z) * s
            wq = torch.where(m, wq, torch.zeros_like(wq))
            qc = torch.where(m, qc, torch.zeros_like(qc))
            Q[:, col] = wq[:, 0]; C[:, col] = qc[:, 0]
            err = (w - wq) / Hinv1[i, i]
            W1[:, i:] -= err @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err[:, 0]
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    if actorder:
        Q = Q[:, invperm]; C = C[:, invperm]
        scale, zero = static
    return C, scale, zero, Q


def dequant(q, scale, zero, col_scale, mask, gsize):
    K, N = q.shape
    ng = N // gsize
    W = (q.view(K, ng, gsize).float() - zero.view(K, ng, 1).float()) * scale.view(K, ng, 1).float()
    return W.view(K, N) * col_scale.view(1, -1) * mask


# ---------------------------------------------------------------- full recipe
def cobalt_quantize(W, H, sparsity, beta, nbits, gsize, hull="survivor",
                    damping_frac=0.01, damping_min=1e-2, mask_block=0,
                    quant_mode="rtn", clip="none", actorder=False, gptq_block=128, H_ho=None):
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
    if quant_mode == "rtn":
        if clip == "none":
            q, scale, zero = group_rtn(W_norm, mask, nbits, gsize, hull)      # shipped path, bit-identical
        else:
            wcol = H.diagonal().clamp(min=0) * c * c                           # ||c_j X_j||^2 in the quantized space
            q, scale, zero = group_quant(W_norm, mask, nbits, gsize, hull, clip, wcol)
    elif quant_mode == "gptq":
        # sequential OBS error feedback in the column-scaled space: W x = (W/c)(c x) => H' = diag(c) H diag(c)
        Hn_c = H * c.view(-1, 1) * c.view(1, -1)
        q, scale, zero, _ = gptq_quantize(W_norm, mask, Hn_c, nbits, gsize, hull, damping_frac, damping_min,
                                          clip=clip, actorder=actorder, blocksize=gptq_block)
        del Hn_c
    else:
        raise ValueError(quant_mode)
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
                 dead_cols=int((mask.sum(0) == 0).sum()), dead_rows=int((mask.sum(1) == 0).sum()),
                 zero_oor=float(((zero < 0) | (zero > 2 ** nbits - 1)).float().mean()))
    if H_ho is not None:
        # HELD-OUT output error on a disjoint calibration realisation (the calibration eout is ~2/3 overfit)
        hs2 = H_ho.diagonal().mean().clamp(min=1e-30)
        Hh = H_ho / hs2
        num2 = float(((D @ Hh) * D).double().sum())
        den2 = float(((W @ Hh) * W).double().sum())
        del Hh
        stats["eout_ho"] = num2 / den2 if den2 > 0 else float("nan")
        stats["eout_ho_num"] = num2; stats["eout_ho_den"] = den2
    return q, scale, zero, c, mask, W_hat, stats
