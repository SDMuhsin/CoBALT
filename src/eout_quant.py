"""E_out-aware survivor requantization (math4 empirical arms).

Implements, inside the unchanged CoBALT pipeline (same mask, same OBS, same
norm scales), three ways to quantize the compensated survivors at STRICTLY
EQUAL TOTAL BITS (bitmap included, per the math4 accounting):

  arm 'rtn'    : the deployed group-RTN (reference; produced by nosink itself).
  arm 'repack' : survivor-only codes at the budget width b' with per-group
                 min-max grids over the SURVIVOR hull (nearest coding).
                 Isolates the "free bits from repacking" effect.
  arm 'eout'   : measured-argmin selector over {D° (deployed output, exact),
                 repack, code-descent(D°), code-descent(repack)} where descent
                 greedily re-codes survivors to minimize the layer-output
                 error ||X (W_hat - W')^T||_F, accepting only strict decreases.
                 Guaranteed <= deployed RTN on the calibration objective by
                 construction (D° is in the candidate set).

Budget accounting (math4 Def. layout): B_codes = b*K*N + 32*(K*N/g - G_ne),
b' = min(B_codes // k, CAP). Codes are conceptually packed survivor-only; for
evaluation we materialize the dequantized fp16 weight (same eval path as every
other method in the harness).

All arms share the deployed storage semantics: per-group (scale, zero) with
the per-row r folded into scales, per-column scale2 = c, decode
((q - z) * s) * c, mask applied post-dequant (pruned positions exactly 0).
"""
import os
import csv
import fcntl

import torch

# fp32 integer-exactness bound is 2^24; keep code values well inside it.
WIDTH_CAP = 20


def _group_view(t, gsize):
    """[K, N] -> [K, n_groups, gsize]; assumes gsize | N (caller checks)."""
    K, N = t.shape
    return t.view(K, N // gsize, gsize)


def budget_width(mask, nbits, gsize):
    """b' = min(floor(B_codes / k), CAP) from the bitmap alone (math4)."""
    K, N = mask.shape
    k = int(mask.sum().item())
    if k == 0:
        return nbits, 0
    if N % gsize != 0:
        return nbits, k  # non-divisible shape: fall back to deployed width
    g_ne = int((_group_view(mask, gsize).sum(-1) > 0).sum().item())
    b_codes = nbits * K * N + 32 * (K * N // gsize - g_ne)
    return max(nbits, min(b_codes // k, WIDTH_CAP)), k


def dequant_deployed(q, scales, zeros, mask, c):
    """Deployed decode: ((q - z) * s) * c, then post-mask. Mirrors
    dequantize_sparse_sinq (r is already folded into scales)."""
    Q = q.float()
    z = zeros.float()
    s = scales.float()
    if s.dim() == 3:
        K, N = Q.shape
        gsize = N // s.shape[1]
        W = ((_group_view(Q, gsize) - z) * s).reshape(K, N)
    else:
        W = (Q - z) * s
    return W * c.view(1, -1) * mask.float()


def repack_quantize(W_norm, mask, bprime, gsize):
    """Per-group min-max RTN over SURVIVOR values of W_norm at width b'.
    Returns (q, scales[K,n_g,1], zeros[K,n_g,1]) in the deployed format
    (deployed clamp min 1e-4 on the range kept for parity)."""
    K, N = W_norm.shape
    m = mask.bool()
    n_levels = 2 ** bprime - 1
    if N > gsize and N % gsize == 0:
        Wg = _group_view(W_norm, gsize)
        Mg = _group_view(m.float(), gsize).bool()
        big = torch.finfo(torch.float32).max
        w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
        w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
        empty = ~Mg.any(-1, keepdim=True)
        w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
        w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
        scales = (w_max - w_min).clamp(min=1e-4) / n_levels
        zeros = -torch.round(w_min / scales)
        q = torch.clamp(torch.round(Wg / scales + zeros), 0, n_levels)
        return q.reshape(K, N), scales, zeros
    # per-row fallback (N <= gsize), mirrors quantize_rtn's use_groups=False
    big = torch.finfo(torch.float32).max
    w_min = torch.where(m, W_norm, torch.full_like(W_norm, big)).amin(1, keepdim=True)
    w_max = torch.where(m, W_norm, torch.full_like(W_norm, -big)).amax(1, keepdim=True)
    empty = ~m.any(1, keepdim=True)
    w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
    w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    scales = (w_max - w_min).clamp(min=1e-4) / n_levels
    zeros = -torch.round(w_min / scales)
    q = torch.clamp(torch.round(W_norm / scales + zeros), 0, n_levels)
    return q, scales, zeros


def eout_sq(W_hat, W_target, H):
    """||X (W_hat - W')^T||_F^2 = tr(D H D^T), fp64 for the verdict."""
    D = (W_hat - W_target).double()
    return float((D @ H.double() * D).sum().item())


# fp16-representable clamp for stored per-group (center, half-range) metadata.
_META_CLAMP = 6e4
# Per-tensor companding exponent grid (1 DOF; <1 = expansive, denser near mode).
GAMMA_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def sign_magnitude_deadzone_quantize(W_norm, mask, nbits, gsize):
    """SIGN-MAGNITUDE + DEAD-ZONE survivor grid. Mechanism: CoBALT's mask removes small
    |w|, so survivors have a near-zero GAP; uniform affine RTN still spends codes across
    [w_min,w_max] INCLUDING that empty central band. Sign-magnitude reclaims them: 1 sign
    + (L/2) magnitude levels placed ONLY on the occupied band [m_lo, m_hi] per group,
    m_lo = min|survivor| (the dead-zone edge). Storage/group = (m_lo, m_hi) = 2 values,
    same as RTN; codes still b bits (encode sign+magnitude); bpw-identical. NOT companding
    (uniform magnitude levels), NOT repack/multiprecision. Non-iterative, global rule.
    Returns W_hat_norm [K,N]."""
    K, N = W_norm.shape
    m = mask.bool()
    L = 2 ** nbits
    half = L // 2                                    # magnitude levels per sign
    grouped = N > gsize and N % gsize == 0
    if grouped:
        Wg = W_norm.view(K, N // gsize, gsize); Mg = m.view(K, N // gsize, gsize)
    else:
        Wg = W_norm.unsqueeze(1); Mg = m.unsqueeze(1)
    A = Wg.abs()
    big = torch.finfo(torch.float32).max
    m_hi = torch.where(Mg, A, torch.zeros_like(A)).amax(-1, keepdim=True).clamp(min=1e-8)
    m_lo = torch.where(Mg, A, torch.full_like(A, big)).amin(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    m_lo = torch.where(empty, torch.zeros_like(m_lo), m_lo)
    span = (m_hi - m_lo).clamp(min=1e-8)
    step = span / max(half - 1, 1)                   # uniform magnitude levels over [m_lo,m_hi]
    lvl = torch.round((A - m_lo) / step).clamp(0, half - 1)
    mag = m_lo + lvl * step
    W_hat = torch.sign(Wg) * mag
    return W_hat.reshape(K, N)


def _compand_grouped(Wg, Mg, gamma, n_levels):
    """Symmetric power-companded RTN over survivors, per group.

    Wg,Mg: [K, ng, gsize] value / bool-mask views. Returns the dequantized
    W_hat_norm [K, ng, gsize] (survivors coded; non-survivors left as Wg — the
    caller re-applies the mask so their value is irrelevant). Storage per group
    = (mu_g, A_g) [same 2 values/group as uniform RTN's (scale, zero)] plus the
    single per-tensor gamma. NO widening (uniform b), NO per-weight fit.

      encode : t = (w - mu_g)/A_g in [-1,1];  v = sign(t)|t|^gamma
               q = round((v+1)/2 * n_levels)              (uniform grid on v)
      decode : v_hat = 2q/n_levels - 1; t_hat = sign|v|^(1/gamma); w = mu_g+A_g t_hat
    """
    big = torch.finfo(torch.float32).max
    cnt = Mg.sum(-1, keepdim=True).clamp(min=1.0)
    mu = torch.where(Mg, Wg, torch.zeros_like(Wg)).sum(-1, keepdim=True) / cnt   # survivor mean
    t0 = Wg - mu
    A = torch.where(Mg, t0.abs(), torch.zeros_like(t0)).amax(-1, keepdim=True).clamp(min=1e-8)
    empty = ~Mg.any(-1, keepdim=True)
    mu = torch.where(empty, torch.zeros_like(mu), mu)
    A = torch.where(empty, torch.ones_like(A), A)
    t = ((Wg - mu) / A).clamp(-1.0, 1.0)
    v = torch.sign(t) * t.abs().pow(gamma)
    q = torch.round((v + 1.0) * 0.5 * n_levels).clamp(0, n_levels)
    v_hat = 2.0 * q / n_levels - 1.0
    t_hat = torch.sign(v_hat) * v_hat.abs().pow(1.0 / gamma)
    return mu + A * t_hat


def compand_quantize(W_norm, mask, nbits, gsize):
    """Fit one per-TENSOR gamma (grid, plain survivor-MSE) then symmetric
    companded-RTN each group. Returns (W_hat_norm [K,N], gamma). Falls back to a
    single per-row group when N<=gsize (mirrors quantize_rtn's use_groups=False).
    A shape prior with 1 DOF/tensor — NOT the per-weight E_out descent (blacklist
    A4). Evaluated held-out downstream, not on this MSE."""
    K, N = W_norm.shape
    m = mask.bool()
    n_levels = 2 ** nbits - 1
    grouped = N > gsize and N % gsize == 0
    if grouped:
        Wg, Mg = _group_view(W_norm, gsize), _group_view(m.float(), gsize).bool()
    else:
        Wg, Mg = W_norm.unsqueeze(1), m.unsqueeze(1)               # [K,1,N]
    best_g, best_e, best_W = 1.0, None, None
    for g in GAMMA_GRID:
        W_hat = _compand_grouped(Wg, Mg, g, n_levels)
        err = float(((W_hat - Wg) * Mg)[Mg].pow(2).sum().item())   # plain survivor MSE
        if best_e is None or err < best_e:
            best_g, best_e, best_W = g, err, W_hat
    return best_W.reshape(K, N), best_g


# binc range-scale alpha (single GLOBAL hyperparameter; env-overridable for sweeps).
# alpha<1 EXPANDS the survivor hull so the extreme survivors are not hard-clipped
# by the bin-centered grid; alpha=0.9 measured output-error-optimal on gemma-2b
# (results/gden_probe: 21/21 matrices beat endpoint-RTN, pooled -11.5%).
BINC_ALPHA = float(os.environ.get("COBALT_BINC_ALPHA", "0.9"))


def bincenter_gated_quantize(W_norm, mask, nbits, gsize, colnorm, alpha=None):
    """ACTIVATION-GATED bin-center: per group, use bin-centered geometry ONLY when the
    group's EXTREME survivor sits in a LOW-activation column (safe to shrink toward the
    interior); otherwise fall back to endpoint-RTN (protect the high-activation extreme).
    Mechanism (measured, results/binc_smoke): binc helps output error when extremes are
    in low-||X|| columns (collapse regime), hurts when extremes carry output energy
    (healthy). A DETERMINISTIC global RULE keyed on ||X|| (already collected for OBS);
    NO per-matrix fit, NO iteration. colnorm: [N] per-column ||X|| (or ||X||^2).
    Returns W_hat_norm [K,N]."""
    alpha = BINC_ALPHA if alpha is None else float(alpha)
    K, N = W_norm.shape
    m = mask.bool()
    L = 2 ** nbits
    grouped = N > gsize and N % gsize == 0
    if grouped:
        Wg = W_norm.view(K, N // gsize, gsize)
        Mg = m.view(K, N // gsize, gsize)
        cg = colnorm.view(1, N // gsize, gsize).expand(K, N // gsize, gsize)
    else:
        Wg = W_norm.unsqueeze(1); Mg = m.unsqueeze(1)
        cg = colnorm.view(1, 1, N).expand(K, 1, N)
    big = torch.finfo(torch.float32).max
    Wm = torch.where(Mg, Wg, torch.zeros_like(Wg))
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
    w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    # locate the extreme (max |w-mid|) survivor's column activation vs group median activation
    mid0 = 0.5 * (w_min + w_max)
    dev = torch.where(Mg, (Wg - mid0).abs(), torch.full_like(Wg, -big))
    ext_idx = dev.argmax(-1, keepdim=True)                       # [K,ng,1]
    ext_colnorm = torch.gather(cg, -1, ext_idx)                  # activation of the extreme's column
    med_colnorm = torch.where(Mg, cg, torch.full_like(cg, big)).median(-1, keepdim=True).values
    gate_binc = (ext_colnorm < med_colnorm)                      # True => shrink is safe => bin-center
    # bin-center branch (alpha-expanded hull)
    mid = mid0
    half = (0.5 * (w_max - w_min)).clamp(min=1e-8) / alpha
    lo = mid - half; step_bc = (2.0 * half) / L
    q_bc = torch.floor((Wg - lo) / step_bc).clamp(0, L - 1)
    W_bc = lo + (q_bc + 0.5) * step_bc
    # endpoint-RTN branch (deployed geometry: L-1 intervals over survivor hull)
    step_ep = (w_max - w_min).clamp(min=1e-4) / (L - 1)
    z_ep = -torch.round(w_min / step_ep)
    q_ep = torch.clamp(torch.round(Wg / step_ep + z_ep), 0, L - 1)
    W_ep = (q_ep - z_ep) * step_ep
    W_hat = torch.where(gate_binc, W_bc, W_ep)
    return W_hat.reshape(K, N)


def bincenter_quantize(W_norm, mask, nbits, gsize, alpha=None):
    """Bin-CENTERED uniform grid over the survivor hull [min,max] per group.
    L=2^b bins (mid-tread, reconstruction at bin CENTERS), hull scaled by alpha
    (alpha<1 expands the hull so extremes are not hard-clipped). MSE-optimal
    uniform quantizer for a UNIFORM bounded source -- the MEASURED CoBALT survivor
    law (kurtosis 1.83 ~ uniform). Same 2 stored values/group as RTN, same dense
    deployed format, uniform b bits (NOT repack/multiprecision/companding).
    Non-iterative, single GLOBAL alpha (no per-layer fit). Returns W_hat_norm[K,N]."""
    alpha = BINC_ALPHA if alpha is None else float(alpha)
    K, N = W_norm.shape
    m = mask.bool()
    L = 2 ** nbits
    grouped = N > gsize and N % gsize == 0
    if grouped:
        Wg = W_norm.view(K, N // gsize, gsize)
        Mg = m.view(K, N // gsize, gsize)
    else:
        Wg = W_norm.unsqueeze(1)
        Mg = m.unsqueeze(1)
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
    w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    mid = 0.5 * (w_min + w_max)
    half = (0.5 * (w_max - w_min)).clamp(min=1e-8) / alpha
    lo = mid - half
    step = (2.0 * half) / L
    q = torch.floor((Wg - lo) / step).clamp(0, L - 1)
    W_hat = lo + (q + 0.5) * step
    return W_hat.reshape(K, N)


# AWCLIP clip-ratio candidate grid (fixed GLOBAL rule; per-group argmin only).
# rho scales the coded half-range: rho<1 clips the (measured ~95% low-energy) group
# extreme, buying finer granularity for the high-energy center survivors.
AWCLIP_RHOS = (1.0, 0.975, 0.95, 0.925, 0.90, 0.875, 0.85)


def _endpoint_rtn_clip_grouped(Wg, Mg, n_levels, rho):
    """Endpoint group-RTN over survivors with the hull half-range scaled by rho.
    Wg,Mg: [K,ng,gsize]. Returns W_hat [K,ng,gsize] (dequantized, normalized space).
    Same (scale,zero) storage as deployed RTN => bpw-identical."""
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
    w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    mid = 0.5 * (w_min + w_max)
    half = (0.5 * (w_max - w_min)).clamp(min=1e-8) * rho
    lo = mid - half
    scale = (2.0 * half) / n_levels
    zero = -torch.round(lo / scale)
    q = torch.clamp(torch.round(Wg / scale + zero), 0, n_levels)
    return (q - zero) * scale


def awclip_quantize(W_norm, mask, nbits, gsize, wcol):
    """ACTIVATION-WEIGHTED per-group SCALE selection. For each group, pick the clip
    ratio rho (from AWCLIP_RHOS, a fixed global grid) that MINIMIZES the group's
    output-error diagonal  sum_{j in g} wcol_j * (W_norm_j - W_hat_j(rho))^2,
    wcol_j = c_j^2 * ||X_j||^2. Deterministic global RULE (no per-matrix fit, no
    iteration, one pass over a fixed candidate set); stores only the resulting per-group
    (scale,zero) = exactly deployed RTN's cost => bpw-identical, same dense decoder.

    Mechanism (measured, src/probe_scale_energy.py): CoBALT's mask keeps the group's
    magnitude-EXTREME in a below-median-||X|| column ~95% of the time, so RTN's max-abs
    scale is dictated by a low-energy weight and its coarse step damages the high-energy
    CENTER survivors that carry the output error. Down-weighting the low-energy extreme
    when choosing the scale (via the wcol weighting) reclaims that granularity WHERE it
    matters, and keeps rho=1 for the minority of groups whose extreme DOES carry energy.
    Returns W_hat_norm [K,N]."""
    K, N = W_norm.shape
    m = mask.bool()
    n_levels = 2 ** nbits - 1
    grouped = N > gsize and N % gsize == 0
    if grouped:
        ng = N // gsize
        Wg = W_norm.view(K, ng, gsize)
        Mg = m.view(K, ng, gsize)
        wcg = wcol.view(1, ng, gsize)
    else:
        Wg = W_norm.unsqueeze(1)
        Mg = m.unsqueeze(1)
        wcg = wcol.view(1, 1, N)
    best_e = None
    best_W = None
    for rho in AWCLIP_RHOS:
        Whn = _endpoint_rtn_clip_grouped(Wg, Mg, n_levels, rho)
        eg = ((Wg - Whn) ** 2 * wcg * Mg).sum(-1, keepdim=True)     # [K,ng,1] weighted err
        if best_e is None:
            best_e = eg
            best_W = Whn
        else:
            take = eg < best_e
            best_e = torch.where(take, eg, best_e)
            best_W = torch.where(take, Whn, best_W)
    return best_W.reshape(K, N)


# AWCLIPZ: joint (scale, zero-point) output-weighted grid. Zero-point is a 2nd already-
# stored fp16 DOF that awclip left at min-max; OBS survivors can be per-group skewed so the
# center shift matters. Fixed global grid (rho half-range x delta center-shift), per-group argmin.
AWCLIPZ_RHOS = (1.0, 0.95, 0.90, 0.85)
AWCLIPZ_DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)


def awclipz_quantize(W_norm, mask, nbits, gsize, wcol):
    """Joint per-group (SCALE, ZERO) selection minimizing the output-error diagonal
    Sum_j wcol_j (W_norm - W_hat)^2 over a fixed global (rho x delta) grid. Same 2 stored
    values/group (scale, zero) as RTN => bpw-identical. Non-iterative, global rule.
    Strictly >= awclip (awclip = the delta=0 slice). Returns W_hat_norm [K,N]."""
    K, N = W_norm.shape
    m = mask.bool()
    n_levels = 2 ** nbits - 1
    grouped = N > gsize and N % gsize == 0
    if grouped:
        ng = N // gsize
        Wg = W_norm.view(K, ng, gsize); Mg = m.view(K, ng, gsize); wcg = wcol.view(1, ng, gsize)
    else:
        Wg = W_norm.unsqueeze(1); Mg = m.unsqueeze(1); wcg = wcol.view(1, 1, N)
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
    w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    mid0 = 0.5 * (w_min + w_max); half0 = (0.5 * (w_max - w_min)).clamp(min=1e-8)
    best_e = None; best_W = None
    for rho in AWCLIPZ_RHOS:
        for dl in AWCLIPZ_DELTAS:
            half = half0 * rho
            lo = (mid0 + dl * half0) - half
            scale = (2.0 * half) / n_levels
            zero = -torch.round(lo / scale)
            q = torch.clamp(torch.round(Wg / scale + zero), 0, n_levels)
            Whn = (q - zero) * scale
            eg = ((Wg - Whn) ** 2 * wcg * Mg).sum(-1, keepdim=True)
            if best_e is None:
                best_e = eg; best_W = Whn
            else:
                take = eg < best_e
                best_e = torch.where(take, eg, best_e); best_W = torch.where(take, Whn, best_W)
    return best_W.reshape(K, N)


def _step_matrix(scales, r, c, K, N, gsize):
    """Per-position decode step a_ij = s_g(i,j) * r_i * c_j (change in W_hat
    per unit code)."""
    if scales.dim() == 3:
        s_full = scales.expand(K, N // gsize, gsize).reshape(K, N)
    else:
        s_full = scales.expand(K, N)
    return s_full * r.view(-1, 1) * c.view(1, -1)


def code_descent(q, scales, zeros, mask, r, c, W_target, H, bprime, gsize,
                 passes=2):
    """Greedy Gauss-Seidel code sweep minimizing tr(D H D^T).

    Columns sequential, rows parallel (rows are independent in the objective).
    Each update solves the exact 1-D quadratic in the integer code, clamps to
    [0, 2^b'-1], and accepts only strict decreases. Grids are FIXED (the codes
    may use the full b' range: step-replication extends deployed grids).
    Returns new q (float tensor of ints)."""
    K, N = W_target.shape
    m = mask.bool()
    n_levels = 2 ** bprime - 1
    a = _step_matrix(scales, r, c, K, N, gsize)          # [K,N]
    q = q.float().clone()
    W_hat = dequant_deployed(q, scales, zeros, mask, c)
    D = (W_hat - W_target).float()                        # [K,N]
    Hf = H.float()
    U = D @ Hf                                            # [K,N] residual grad
    hdiag = Hf.diag()                                     # [N]
    for _ in range(passes):
        improved = False
        for j in range(N):
            mj = m[:, j]
            if not mj.any():
                continue
            aj = a[:, j]
            hjj = hdiag[j]
            denom = aj * aj * hjj
            # exact 1-D quadratic: dE(dq) = 2*aj*dq*U[:,j] + aj^2*hjj*dq^2
            dq_star = torch.where(denom > 0, -aj * U[:, j] / denom.clamp(min=1e-30),
                                  torch.zeros_like(aj))
            q_new = torch.clamp(torch.round(q[:, j] + dq_star), 0, n_levels)
            dq = q_new - q[:, j]
            dE = 2.0 * aj * dq * U[:, j] + denom * dq * dq
            take = mj & (dE < 0) & (dq != 0)
            if not take.any():
                continue
            improved = True
            dw = torch.where(take, aj * dq, torch.zeros_like(aj))    # [K]
            q[:, j] = torch.where(take, q_new, q[:, j])
            D[:, j] = D[:, j] + dw
            U += dw.view(-1, 1) * Hf[j].view(1, -1)
        if not improved:
            break
    return q


def eout_requantize(W_comp, mask, X, nbits, gsize, r, c,
                    q_dep, scales_dep, zeros_dep, arm, passes=2):
    """Produce the arm's dequantized weight [K,N] fp32 + info dict.

    W_comp: OBS-compensated target W' (fp32, zeros off-support).
    X: calibration activations [T, N] (already flattened/capped by caller).
    q_dep/scales_dep/zeros_dep: the deployed RTN's stored tensors (r folded
    into scales_dep already, as nosink does).
    """
    K, N = W_comp.shape
    Wt = W_comp.float()
    mk = mask.float()
    H = (X.float().t() @ X.float())                       # [N,N]

    W_rtn = dequant_deployed(q_dep, scales_dep, zeros_dep, mk, c)
    e_rtn = eout_sq(W_rtn, Wt, H)
    bprime, k = budget_width(mk, nbits, gsize)
    info = {"bprime": int(bprime), "k": int(k), "e_rtn": e_rtn}

    if arm == "rtn" or k == 0 or (N % gsize != 0 and N > gsize):
        info.update({"e_repack": e_rtn, "e_arm": e_rtn, "picked": "rtn"})
        return W_rtn, info

    if arm == "bincg":
        # ACTIVATION-GATED binc: per-group endpoint-vs-bincenter by the extreme survivor's
        # column activation. Deterministic global rule, no fit/iteration. bpw == RTN.
        W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
        colnorm = (X.float() * X.float()).sum(0).clamp(min=0)      # [N] ||X_j||^2
        W_hat_n = bincenter_gated_quantize(W_norm, mk, nbits, gsize, colnorm)
        W_bg = W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * mk
        e_bg = eout_sq(W_bg, Wt, H)
        info.update({"e_repack": e_rtn, "e_arm": e_bg, "picked": "bincg",
                     "alpha": BINC_ALPHA, "bprime": nbits})
        return W_bg, info

    if arm == "binc":
        # NOVEL: bin-CENTERED uniform survivor grid (alpha-scaled hull) fit to
        # CoBALT's MEASURED near-uniform survivor law. Same 2 vals/group as RTN,
        # uniform b bits, dense format => bpw-identical to deployed cobalt-rtn.
        W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
        W_hat_n = bincenter_quantize(W_norm, mk, nbits, gsize)
        W_bc = W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * mk
        e_bc = eout_sq(W_bc, Wt, H)
        info.update({"e_repack": e_rtn, "e_arm": e_bc, "picked": "binc",
                     "alpha": BINC_ALPHA, "bprime": nbits})
        return W_bc, info

    if arm == "awclip":
        # NOVEL (attempt-7): activation-weighted per-group SCALE. Same dense (scale,zero)
        # decoder as deployed RTN => bpw-identical; only the per-group scale is chosen to
        # minimize the OUTPUT-error diagonal instead of max-abs. Targets tr(DHD^T), not the
        # (near-uniform) marginal value law — the axis every prior arm left at RTN's max-abs.
        W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
        colE = (X.float() * X.float()).sum(0).clamp(min=0)          # [N] ||X_j||^2
        wcol = (c.float() ** 2) * colE                              # diag output weight (norm space)
        W_hat_n = awclip_quantize(W_norm, mk, nbits, gsize, wcol)
        W_ac = W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * mk
        e_ac = eout_sq(W_ac, Wt, H)
        info.update({"e_repack": e_rtn, "e_arm": e_ac, "picked": "awclip", "bprime": nbits})
        return W_ac, info

    if arm == "awclipz":
        W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
        colE = (X.float() * X.float()).sum(0).clamp(min=0)
        wcol = (c.float() ** 2) * colE
        W_hat_n = awclipz_quantize(W_norm, mk, nbits, gsize, wcol)
        W_az = W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * mk
        e_az = eout_sq(W_az, Wt, H)
        info.update({"e_repack": e_rtn, "e_arm": e_az, "picked": "awclipz", "bprime": nbits})
        return W_az, info

    if arm == "compand":
        # NOVEL: uniform-width companded survivor grid fit to CoBALT's post-OBS
        # shape (1 gamma/tensor, storage = same 2 vals/group as RTN). Decode to
        # W_comp space exactly as dequant_deployed (x r x c, post-mask).
        W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
        W_hat_n, gamma = compand_quantize(W_norm, mk, nbits, gsize)
        W_cp = W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * mk
        e_cp = eout_sq(W_cp, Wt, H)
        info.update({"e_repack": e_rtn, "e_arm": e_cp, "picked": "compand",
                     "gamma": gamma, "bprime": nbits})
        return W_cp, info

    # ---- repack arm: survivor-hull min-max grids at b', nearest coding.
    # Work in the same normalized space the deployed RTN quantizes, then fold
    # r into scales exactly as nosink does, so decode semantics are identical.
    W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
    q_rp, s_rp, z_rp = repack_quantize(W_norm, mk, bprime, gsize)
    if s_rp.dim() == 3:
        s_rp = s_rp * r.view(-1, 1, 1)
    else:
        s_rp = s_rp * r.view(-1, 1)
    s_rp = torch.nan_to_num(s_rp, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
    W_rp = dequant_deployed(q_rp, s_rp, z_rp, mk, c)
    e_rp = eout_sq(W_rp, Wt, H)
    info["e_repack"] = e_rp

    if arm == "repack":
        info.update({"e_arm": e_rp, "picked": "repack"})
        return W_rp, info

    assert arm == "eout"
    cands = [("rtn", W_rtn, e_rtn), ("repack", W_rp, e_rp)]
    # descent from D° (deployed grids, code range extended to b' by step
    # replication — exactly math4's D° lever) and from repack grids.
    q_d1 = code_descent(q_dep.float(), scales_dep.float(), zeros_dep.float(),
                        mk, r, c, Wt, H, bprime, gsize, passes=passes)
    W_d1 = dequant_deployed(q_d1, scales_dep.float(), zeros_dep.float(), mk, c)
    cands.append(("descend-rtn", W_d1, eout_sq(W_d1, Wt, H)))
    q_d2 = code_descent(q_rp, s_rp, z_rp, mk, r, c, Wt, H, bprime, gsize,
                        passes=passes)
    W_d2 = dequant_deployed(q_d2, s_rp, z_rp, mk, c)
    cands.append(("descend-repack", W_d2, eout_sq(W_d2, Wt, H)))
    # selector: measured argmin; ties/NaNs fall back to earliest (rtn first)
    best = min(range(len(cands)),
               key=lambda i: (cands[i][2] if cands[i][2] == cands[i][2] else float("inf"), i))
    name, W_best, e_best = cands[best]
    for nm, _, ev in cands:
        info[f"e_{nm.replace('-', '_')}"] = ev
    info.update({"e_arm": e_best, "picked": name})
    return W_best, info


def repack_only(W_target, mask, nbits, gsize, r, c):
    """Method-agnostic repack: survivor-only b'-bit codes with survivor-hull
    min-max grids, in the caller's normalized space W_target/(r (x) c).

    Every pruned+quant method supplies its own (mask, r, c, W_target); the
    ONLY thing that changes across methods is the normalization (r, c) and the
    survivor values, never the storage lever. Returns (W_dense[K,N], info).

      cobalt : r=1(folded), c=col-scale,  W_target=W_comp (OBS)
      wanda-awq  : r=1,     c=1/awq_scale, W_target=W_pruned
      wanda-sinq : r=mu2,   c=mu1,         W_target=W_pruned  (Sinkhorn dual)
    """
    K, N = W_target.shape
    Wt = W_target.float()
    mk = mask.float()
    bprime, k = budget_width(mk, nbits, gsize)
    info = {"bprime": int(bprime), "k": int(k)}
    if k == 0 or (N % gsize != 0 and N > gsize):
        return Wt * mk, {**info, "picked": "empty"}
    W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
    q, s, z = repack_quantize(W_norm, mk, bprime, gsize)
    if s.dim() == 3:
        s = s * r.view(-1, 1, 1)
    else:
        s = s * r.view(-1, 1)
    s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
    W_dense = dequant_deployed(q, s, z, mk, c)
    return W_dense, {**info, "picked": "repack"}


def bincenter_only(W_target, mask, nbits, gsize, r, c, alpha=None):
    """Method-agnostic BINC: bin-centered uniform survivor grid (alpha-scaled hull) in
    the caller's normalized space W_target/(r (x) c), decoded back and post-masked.
    Mirrors `repack_only` so any pruned+quant method gets the SAME grid-geometry lever
    on ITS OWN normalization (fairness: binc is mask-agnostic, so baselines get it too).
      cobalt     : r=1(folded), c=col-scale, W_target=W_comp (OBS)
      wanda-awq  : r=1,        c=1/awq_scale, W_target=W_pruned
      wanda-sinq : r=mu2,      c=mu1,         W_target=W_pruned
    Same storage as RTN (2 vals/group), uniform b bits ⇒ bpw-identical. Returns W_dense[K,N]."""
    K, N = W_target.shape
    Wt = W_target.float()
    mk = mask.float()
    k = int(mk.sum().item())
    if k == 0 or (N % gsize != 0 and N > gsize):
        return Wt * mk
    W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
    W_hat_n = bincenter_quantize(W_norm, mk, nbits, gsize, alpha=alpha)
    return W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * mk


def awclip_only(W_target, mask, nbits, gsize, r, c, colE, alpha=None):
    """Method-agnostic AWCLIP: activation-weighted per-group SCALE selection in the
    caller's normalized space W_target/(r (x) c), decoded back and post-masked. Mirrors
    `repack_only`/`bincenter_only` so ANY pruned+quant method gets the SAME scale-selection
    lever on ITS OWN normalization (fairness: awclip is mask-agnostic, so baselines get it
    too — the repack fairness rule).  colE = [N] ||X_j||^2 in the ORIGINAL input space.
      cobalt     : r=1(folded), c=col-scale,  W_target=W_comp (OBS)
      wanda-awq  : r=1,        c=1/awq_scale,  W_target=W_pruned
      wanda-sinq : r=mu2,      c=mu1,          W_target=W_pruned
    Same storage as RTN (2 vals/group), uniform b bits => bpw-identical. Returns W_dense[K,N]."""
    K, N = W_target.shape
    Wt = W_target.float()
    mk = mask.float()
    k = int(mk.sum().item())
    if k == 0 or (N % gsize != 0 and N > gsize):
        return Wt * mk
    W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
    wcol = (c.float() ** 2) * colE.float()                     # diag output weight in norm space
    W_hat_n = awclip_quantize(W_norm, mk, nbits, gsize, wcol)
    return W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * mk


def awclipz_only(W_target, mask, nbits, gsize, r, c, colE):
    """Method-agnostic AWCLIPZ (joint scale+zero output-weighted grid) in the caller's
    normalized space W_target/(r (x) c). Mirrors awclip_only so baselines get the SAME
    lever (fairness). Same storage as RTN => bpw-identical. Returns W_dense[K,N]."""
    K, N = W_target.shape
    Wt = W_target.float(); mk = mask.float()
    k = int(mk.sum().item())
    if k == 0 or (N % gsize != 0 and N > gsize):
        return Wt * mk
    W_norm = Wt / (r.view(-1, 1) * c.view(1, -1))
    wcol = (c.float() ** 2) * colE.float()
    W_hat_n = awclipz_quantize(W_norm, mk, nbits, gsize, wcol)
    return W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * mk


def log_margin(path, row):
    """Append one per-layer margin row (flock'd, header-once)."""
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["model", "method", "sparsity", "bits", "layer", "type",
              "bprime", "k", "e_rtn", "e_repack", "e_arm", "picked",
              "rel_vs_rtn"]
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)
