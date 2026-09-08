#!/usr/bin/env python3
"""NON-SINKHORN joint PTQ base: Wanda-mask + OBS compensation + group-RTN.

Req #3 forbids Sinkhorn. This module implements the quantizer WITHOUT any
`sinkhorn_log` call: the pruning mask is plain Wanda importance |W|·‖X‖ (no μ),
error is redistributed by OBS (Hessian inverse, non-Sinkhorn), survivors are
group-RTN quantized, and the dual-scale μ1/μ2 are FIXED to 1 (no normalization).
Reuses only `compute_hessian_inverse` and `quantize_rtn` from sinq.sparse_quant
(both Sinkhorn-free) and `SparseQuantLinear` for storage/dequant.

Calibration cost ≤ PRISM: same 8 forward passes + same OBS Hessian inverse, but
DROPS PRISM's 64 Sinkhorn iterations. Overhead identical to PRISM (group-64 fp16
scale+zero = 0.5 bpw; scale2=1 costs nothing).

Design lever (from measurement): the GQA value bottleneck v_proj cannot tolerate
70% pruning (its 3-bit quant is free). `--mode vdense` allocates 0% sparsity to
v_proj and (with --hold-global) prunes other types slightly more so GLOBAL
sparsity stays EXACTLY 0.70 — honest, no bit-budget cheat.
"""
import argparse
import csv
import gc
import os
import sys

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa: E402
from sinq.sparse_quant import compute_hessian_inverse, quantize_rtn  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from tqdm import tqdm  # noqa: E402

# Diagnostic-only import (Sinkhorn upper bound; NOT used by any deliverable mode).
try:
    from sinq.sinkhorn import sinkhorn_log as _sinkhorn_log  # noqa: E402
except Exception:  # pragma: no cover
    _sinkhorn_log = None
try:
    from sinq.sparse_quant import sinkhorn_log_sparse_aware as _sinkhorn_sa  # noqa: E402
except Exception:  # pragma: no cover
    _sinkhorn_sa = None

MODEL = "gemma-2b"
NBITS = 3
GROUP_SIZE = 64
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]


def type_of(attr_path: str):
    suffix = attr_path.split('.')[-1]
    return suffix[:-len('_proj')] if suffix.endswith('_proj') else suffix


# N:M semi-structured switch (2:4 route). When set to (n, m), EVERY mask produced by this module keeps
# exactly the top-n of each m consecutive INPUT entries (along dim=1) of the importance that reaches
# _threshold_mask -- i.e. CoBALT's row+col quantile reweighting still shapes WHICH n survive (the column
# term is the only live DOF: a per-row rescale cannot reorder entries inside a row's own m-group), but
# the support is hardware-realizable (Sparse-Marlin / sparse tensor cores). Sparsity must equal 1-n/m.
NM_PATTERN = None


def nm_mask(importance, n, m):
    """Keep-mask (float, 1=keep) with exactly n survivors per m consecutive entries along dim=1."""
    K, N = importance.shape
    assert N % m == 0, f"N={N} not divisible by m={m} for {n}:{m} sparsity"
    g = importance.view(K, N // m, m)
    idx = g.topk(n, dim=-1).indices
    mask = torch.zeros_like(g)
    mask.scatter_(-1, idx, 1.0)
    return mask.view(K, N)


def _threshold_mask(importance, sparsity, scope='global'):
    """Binarize an importance map to a keep-mask at the given sparsity.
    scope='global': single GLOBAL top-k threshold over ALL K*N entries (PRISM's
      convention; can starve low-scale rows — M1 shows up to ~22% of rows nearly
      fully pruned unless the importance is scale-invariant).
    scope='per_row': a SEPARATE threshold per output row so EVERY row keeps exactly
      (1-sparsity) of its entries — guarantees no dead output channel by construction
      (the secondary FINDINGS hypothesis for beating PRISM's global mask)."""
    K, N = importance.shape
    if sparsity <= 0.0:
        return torch.ones_like(importance)
    if NM_PATTERN is not None:                             # 2:4 route: N:M top-n per m-group
        return nm_mask(importance, *NM_PATTERN)
    if scope == 'per_row':
        k_prune = int(N * sparsity)
        if k_prune <= 0:
            return torch.ones_like(importance)
        thr = torch.kthvalue(importance, k_prune, dim=1, keepdim=True).values  # [K,1]
        return (importance > thr).float
    n_prune = int(K * N * sparsity)
    flat = importance.view(-1)
    thr = torch.kthvalue(flat, n_prune).values
    return (flat > thr).view(K, N).float


def wanda_mask_and_obs(W, X, sparsity, device, scope='global', act_exp=1.0):
    """Wanda mask (|W|·‖X‖^act_exp, scope global or per_row) + OBS compensation. NO Sinkhorn.
    act_exp>1 boosts the activation exponent: the μ1-proxy diagnostic shows PRISM's inverse-μ
    μ1 ANTI-correlates with ‖X‖, so importance=|W|·‖X‖/μ1 ≈ |W|·‖X‖^(1+β) — i.e. the Sinkhorn
    mask effectively raises the ‖X‖ exponent. This is the closed-form, non-Sinkhorn analog.
    Returns (W_compensated, mask). Wanda+OBS path mirrors sparse_with_prism, no sinkhorn_log."""
    K, N = W.shape
    W = W.float.to(device)
    if sparsity <= 0.0:
        return W.clone, torch.ones(K, N, device=device)
    X = X.float.to(device)
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N], L2 per input channel
    importance = W.abs * act_norms.view(1, -1).pow(act_exp)  # Wanda, activation exponent
    mask = _threshold_mask(importance, sparsity, scope)
    # OBS compensation (Hessian only; Sinkhorn-free)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag
    W_comp = W.clone
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def balanced_keepmask_local(imp, sparsity, col_exp):
    """CoBALT balanced keep-mask (row quantile + col^col_exp quantile + global top-k) from an
    importance map imp[K,N]. Standalone version used by candidate-generating levers (balanced_stoch)."""
    K, N = imp.shape
    kr, kc = int(N * sparsity), int(K * sparsity)
    imp = imp.clone
    if kr > 0:
        imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
    if kc > 0:
        imp = imp / torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30).pow(col_exp)
    n_prune = int(K * N * sparsity)
    thr = torch.kthvalue(imp.reshape(-1), n_prune).values
    return (imp.reshape(-1) > thr).view(K, N).float


def column_floor_mask(base_mask, imp, sparsity, floor_frac):
    """HARD column-degree-floor operator (attempt-11, backlog #21). Takes any base keep-mask and its
    ranking importance `imp` [K,N] (higher = more important), and enforces a HARD minimum survivor
    count per COLUMN f = round(floor_frac*(1-sp)*K), WHILE HOLDING THE GLOBAL SURVIVOR COUNT EXACTLY
    CONSTANT (so effective bpw is identical). For each under-floor column, PROMOTE its top-`imp` pruned
    entries up to f; DEMOTE the exactly-equal number of globally-weakest survivors, taken only from
    columns that stay >= f (per-column surplus cap => floor never violated by the demotion).

    Mechanism (MEASURED, src/probe_colfloor.py): the deployed soft-beta balanced mask still leaves
    ~0.1-2% DEAD columns + 3-5% under-half at sp0.7 in healthy models -- a dead input channel is a
    categorical, OBS-uncompensable information loss. This rescues exactly those columns. ONE-SHOT
    (single promote/demote selection, like a top-k), NON-separable (per-column degree constraint
    couples rows<->cols; survives the row-absorption result), no per-layer error opt, no iteration.
    floor_frac=0 -> identity; floor_frac=1 -> exact per-column balance. Returns new keep-mask [K,N]."""
    if floor_frac <= 0.0 or sparsity <= 0.0:
        return base_mask
    K, N = base_mask.shape
    dev = base_mask.device
    f = int(round(floor_frac * (1.0 - sparsity) * K))
    if f <= 0:
        return base_mask
    surv = base_mask.bool
    col_cnt = surv.sum(dim=0)                                  # [N] survivors per column
    deficit = (f - col_cnt).clamp(min=0)                       # [N] to add per column
    total_add = int(deficit.sum.item)
    if total_add == 0:
        return base_mask
    NEG, POS = float('-inf'), float('inf')
    # --- PROMOTION: per deficit column, the top-`imp` PRUNED entries (survivors set to -inf) ---
    imp_p = torch.where(surv, torch.full_like(imp, NEG), imp)  # only pruned entries eligible
    idx_desc = imp_p.argsort(dim=0, descending=True)           # [K,N] rows sorted by desc imp per col
    take = (torch.arange(K, device=dev).view(K, 1) < deficit.view(1, N))  # top deficit_c positions
    promote = torch.zeros_like(surv)
    promote.scatter_(0, idx_desc, take)                        # scatter back to original positions
    # --- DEMOTION: exactly total_add globally-weakest survivors, capped by per-column surplus ---
    surplus = (col_cnt - f).clamp(min=0)                       # [N] removable budget per column
    imp_s = torch.where(surv, imp, torch.full_like(imp, POS))  # only survivors eligible, ascending
    asc = imp_s.argsort(dim=0)                                 # [K,N] rows sorted by asc imp per col
    within = (torch.arange(K, device=dev).view(K, 1) < surplus.view(1, N))  # weakest `surplus_c` per col
    removable = torch.zeros_like(surv)
    removable.scatter_(0, asc, within)                         # removable survivors (floor-safe)
    imp_flat = imp.reshape(-1).clone
    imp_flat[~removable.reshape(-1)] = POS
    order = torch.argsort(imp_flat)                            # ascending; weakest removable first
    demote_idx = order[:total_add]
    demote = torch.zeros(K * N, dtype=torch.bool, device=dev)
    demote[demote_idx] = True
    demote = demote.view(K, N)
    new_mask = surv.clone
    new_mask[promote] = True
    new_mask[demote] = False
    return new_mask.float


def exact_doubly_balanced_mask(imp, sparsity):
    """EXACT doubly-degree-balanced keep-mask (attempt-11 #18). One-shot, non-iterative. Greedy: sort
    entries by importance desc, keep if BOTH its row and column are below their exact keep-caps
    kr=round((1-sp)*N), kc=round((1-sp)*K); then fill under-filled ROWS (col-cap relaxed) to hit EXACT
    per-row keep-rate => exact global sparsity. Enforces HARD column balance (measured keep-rate CV ~4x
    lower than soft-beta, Jaccard ~0.82 => a real, non-absorbed lever) while RETAINING magnitude info
    (ranks by |W|.||X||, unlike rank-sum). Sharpens the collapse-rescue balance mechanism to its hard
    form. Returns keep-mask [K,N] float."""
    K, N = imp.shape
    kr = int(round((1.0 - sparsity) * N))
    kc = int(round((1.0 - sparsity) * K))
    if kr <= 0:
        return torch.zeros_like(imp)
    dev = imp.device
    order = torch.argsort(imp.reshape(-1), descending=True)
    od = order.to('cpu').numpy
    ro = (order // N).to('cpu').numpy; co = (order % N).to('cpu').numpy
    rc = [0] * K; cc = [0] * N
    keepl = bytearray(K * N)
    full_rows = 0
    for t in range(len(od)):
        i = int(ro[t]); j = int(co[t])
        if rc[i] < kr and cc[j] < kc:
            keepl[od[t]] = 1; rc[i] += 1; cc[j] += 1
            if rc[i] == kr:
                full_rows += 1
                if full_rows == K:            # all rows satisfied -> exact global sparsity reached
                    break
    if full_rows < K:                          # PASS 2: fill rows blocked by col caps (col-cap relaxed)
        for t in range(len(od)):
            i = int(ro[t])
            if rc[i] < kr and keepl[od[t]] == 0:
                keepl[od[t]] = 1; rc[i] += 1
    return torch.tensor(list(keepl), dtype=torch.float32, device=dev).view(K, N)


def balanced_mask_and_obs(W, X, sparsity, device, col_exp=1.0, row_fair=True, per_row_thresh=False,
                          no_obs=False, imp_base=None):
    """COLUMN-BALANCED Wanda mask + OBS. NON-Sinkhorn, motivated by a DIRECT measurement
    (diag_column_balance): inverse-μ's downstream win = it keeps a COLUMN-BALANCED survivor set
    (lowest per-column keep-CV + fewest dead columns in EVERY matrix), whereas saliency masks
    (Wanda, exact-Fisher) STARVE columns (concentrate survivors in high-|W|·‖X‖ columns). Every
    saliency criterion (reconstruction, sensitivity, Fisher) fails downstream for this reason; the
    lever is BALANCE, not saliency. This enforces balance DIRECTLY (no Sinkhorn, no μ1-value
    replication which was shown to resist closed form): self-normalize each ROW and each COLUMN of
    the importance by its own (1-sp) quantile so keep-rates are ~uniform along both axes, then a
    single GLOBAL top-k. ONE-SHOT (Sinkhorn ITERATES this to convergence — forbidden; one pass is
    not Sinkhorn). col_exp=β scales the column-balancing strength: β=0 ⇒ per-row Wanda (F, rows only);
    β=1 ⇒ full column balance. Returns (W_compensated, mask)."""
    K, N = W.shape
    W = W.float.to(device)
    if sparsity <= 0.0:
        return W.clone, torch.ones(K, N, device=device)
    X = X.float.to(device)
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N]
    if imp_base is not None:
        # WITHIN-matrix INTERACTION lever (balanced_fint): replace the marginal |W|.||X|| importance
        # with |W|.sqrt(E[g_i^2 x_j^2]) -- the NON-separable g-x correlation is the only signal that can
        # move the balanced survivor set (src/probe_within_mask.py row-absorption result). Same balanced
        # row+col quantile thresholding + same OBS below.
        imp = imp_base.to(device).float
    else:
        imp = W.abs * act_norms.view(1, -1)              # Wanda base
    kr, kc = int(N * sparsity), int(K * sparsity)
    if per_row_thresh:
        # UNIMPEACHABLY non-Sinkhorn variant: a SINGLE per-column reweighting (like AWQ's per-column
        # activation scaling — universally accepted as non-Sinkhorn) + a PER-ROW Wanda threshold.
        # Zero row/column alternation. Column balance comes purely from downweighting over-represented
        # columns (by their (1-sp) quantile) so they win fewer per-row slots; rows are exact by the
        # per-row threshold. No global budget, no row-normalization pass.
        if col_exp > 0 and kc > 0:
            qc = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)  # [1,N]
            imp = imp / qc.pow(col_exp)
        mask = _threshold_mask(imp, sparsity, scope='per_row')
    else:
        if row_fair and kr > 0:                            # row self-normalization (per-row balance)
            qr = torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)  # [K,1]
            imp = imp / qr
        if col_exp > 0 and kc > 0:                         # column self-normalization (the NEW lever)
            qc = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)  # [1,N]
            imp = imp / qc.pow(col_exp)
        mask = _threshold_mask(imp, sparsity, scope='global')  # shared budget ⇒ balance is meaningful
    if no_obs:
        # ABLATION Factor C = OFF: drop OBS error-redistribution entirely. The mask is computed
        # IDENTICALLY above (byte-identical for a given col_exp) so this is orthogonal to Factor M;
        # survivors are simply kept uncompensated and handed to the same downstream group-RTN.
        return W * mask, mask
    # OBS compensation (identical to every other recipe)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag
    W_comp = W.clone
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def _grid_swap_support(W_norm, mask, energy, gsize):
    """QUANTIZATION-GRID-AWARE support swap (deterministic global rule, non-iterative).
    Per group: the scale-setter s* = survivor with max |w_norm - group_mid| sets the group-RTN
    step. If s* sits in a BELOW-median-ENERGY position (energy = c^2*||X||^2; measured ~95% of the
    time under CoBALT's mask), PRUNE it and PROMOTE the highest-ENERGY pruned candidate that is
    INTERIOR to the shrunken hull (|w-mid| <= 2nd-largest deviation, so it does NOT re-extend the
    scale). Holds per-group survivor count => global sparsity unchanged; shrinks the hull => finer
    step for all group-mates. MEASURED (src/probe_gridmask.py): reduces post-quant output error at
    2-bit (coarse grid, hull dominates) CoBALT-specifically; NEUTRAL/harmful >=3-bit (do not use).
    Returns the new boolean keep-mask [K,N] (float)."""
    K, N = W_norm.shape
    m = mask.bool
    if not (N > gsize and N % gsize == 0):
        return mask
    ng = N // gsize
    Wg = W_norm.view(K, ng, gsize); Mg = m.view(K, ng, gsize)
    Eg = energy.view(1, ng, gsize).expand(K, ng, gsize)
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    valid = Mg.any(-1, keepdim=True)
    mid = 0.5 * (w_min + w_max)
    dev = torch.where(Mg, (Wg - mid).abs, torch.full_like(Wg, -big))
    top2 = dev.topk(2, dim=-1).values
    s_dev = top2[..., 0:1]; new_half = top2[..., 1:2].clamp(min=0)
    s_idx = dev.argmax(-1, keepdim=True)
    ext_E = torch.gather(Eg, -1, s_idx)
    med_E = torch.where(Mg, Eg, torch.full_like(Eg, big)).median(-1, keepdim=True).values
    dev_p = (Wg - mid).abs
    interior = (~Mg) & (dev_p <= new_half)
    cand_E = torch.where(interior, Eg, torch.full_like(Eg, -big))
    p_idx = cand_E.argmax(-1, keepdim=True)
    fire = (valid & interior.any(-1, keepdim=True) & (s_dev > 0) & (ext_E < med_E)).squeeze(-1)
    Mg_new = Mg.clone
    ar = torch.arange(K, device=W_norm.device).view(-1, 1).expand(K, ng)[fire]
    ag = torch.arange(ng, device=W_norm.device).view(1, -1).expand(K, ng)[fire]
    Mg_new[ar, ag, s_idx.squeeze(-1)[fire]] = False
    Mg_new[ar, ag, p_idx.squeeze(-1)[fire]] = True
    return Mg_new.reshape(K, N).float


def grid_balanced_mask_and_obs(W, X, sparsity, device, col_exp=1.0, gsize=None, no_obs=False):
    """GRID-AWARE column-balanced mask + OBS. Two-pass, NON-iterative, NON-Sinkhorn:
    (1) the deployed CoBALT balanced mask + OBS + col-scale c (via balanced_mask_and_obs);
    (2) a deterministic GRID-AWARE support swap (_grid_swap_support) that prunes each group's
        low-energy scale-setter and promotes an interior high-energy candidate, tightening the
        group-RTN hull; then RE-OBS on the new support. Motivated by the MEASURED coupling that
        CoBALT's mask parks the group magnitude-extreme in a low-||X|| column (src/probe_gridmask,
        src/probe_scale_energy). Deployable ONLY at low bit-width (2-bit); >=3-bit the survivor-loss
        cost exceeds the granularity gain (measured). Returns (W_compensated, mask)."""
    gsize = GROUP_SIZE if gsize is None else int(gsize)
    K, N = W.shape
    if sparsity <= 0.0:
        Wf = W.float.to(device)
        return Wf.clone, torch.ones(K, N, device=device)
    # pass 1: balanced mask + OBS + col-scale
    W_comp0, mask0 = balanced_mask_and_obs(W, X, sparsity, device, col_exp=col_exp)
    _, c0 = compute_norm_scales(W_comp0, mask0, 'col', device)
    W_norm0 = W_comp0 / c0.view(1, -1)
    Xf = X.float.to(device)
    if Xf.dim == 3:
        Xf = Xf.reshape(-1, Xf.shape[-1])
    Xf = Xf[:min(Xf.shape[0], 256)]
    colE = (Xf * Xf).sum(0).clamp(min=0)                        # ||X_j||^2 [N]
    energy_norm = (c0.float ** 2) * colE                      # diag output weight in norm space
    mask1 = _grid_swap_support(W_norm0, mask0, energy_norm, gsize)
    Wf = W.float.to(device)
    if no_obs:
        return Wf * mask1, mask1
    # pass 2: re-OBS on the swapped support (identical one-shot batch OBS, vectorized)
    H_inv = compute_hessian_inverse(Xf, damping=None)
    H_inv_diag = H_inv.diag
    P = (Wf * (1.0 - mask1)) / H_inv_diag.view(1, -1)
    W_comp1 = (Wf - P @ H_inv) * mask1
    return W_comp1, mask1


def sens_wanda_mask_and_obs(W, X, sparsity, device, scope='global', sens_row=None, sens_exp=0.5,
                            fair=False):
    """G-AWARE (output-Fisher) Wanda mask + OBS. NON-Sinkhorn, PRISM-orthogonal.

    First-principles: to 2nd order the task-loss increase of a linear layer is
    ½ tr(ΔWᵀ H_act ΔW G), G = E[g gᵀ] the output-gradient covariance (K-FAC). Under the
    measured-robust diagonal approximations H_act≈diag(‖X‖²) (Wanda; correlation-aware OBS
    overfits at S=256 — diag_mask_quality) and G≈diag(s), s_i=E[(∂L/∂y_i)²]:
        ΔL ≈ ½ Σ_i s_i Σ_j ‖X‖_j² Δw_ij²
    ⇒ per-ENTRY pruning saliency = s_i·(‖X‖_j·|w_ij|)²  = Wanda saliency scaled per-row by the
    output-channel SENSITIVITY s_i. A single global top-k on it is the EXACT water-filling
    allocation of survivors under the diagonal task-loss objective; s_i is constant within a
    row so it changes only CROSS-ROW allocation (within-row selection stays Wanda). This is
    the untried, PRISM-orthogonal output-side lever (PRISM uses only input ‖X‖ + weight-μ;
    reconstruction sets G=I). importance = |W|·‖X‖·s_i^sens_exp (sens_exp=0.5 ⇔ theory-exact
    on the squared saliency; 0 ⇔ plain Wanda).

    fair=True (ROW-FAIR allocation, GLOBAL scope): plain global Wanda starves ~22% of rows via
    a WEIGHT-SCALE artifact (M1) independent of the loss — global sens-Wanda inherits it, and a
    per-row threshold cures it but makes s_i INERT (constant within a row). To isolate the
    sensitivity lever we first make the cross-row comparison FAIR by normalizing each row's
    saliency by its own (1-sp)-quantile q_i (so at sens_exp=0 the global top-k keeps ≈(1-sp) per
    row = per-row Wanda, no scale artifact), THEN tilt the shared budget by s_i^sens_exp. Net:
    importance = s_i^sens_exp · (|W|·‖X‖) / q_i, global top-k. sens_exp=0 ⇒ per-row Wanda;
    sens_exp>0 ⇒ high-sensitivity rows keep MORE, low keep fewer — pure loss-driven reallocation
    on a fair baseline. Closed-form (one kthvalue/row), non-Sinkhorn, < PRISM calibration.
    Returns (W_compensated, mask)."""
    K, N = W.shape
    W = W.float.to(device)
    if sparsity <= 0.0:
        return W.clone, torch.ones(K, N, device=device)
    X = X.float.to(device)
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N], L2 per input channel
    importance = W.abs * act_norms.view(1, -1)           # Wanda base saliency
    if fair:                                               # row-fair: normalize by per-row (1-sp) quantile
        k_prune = int(N * sparsity)
        if k_prune > 0:
            q = torch.kthvalue(importance, k_prune, dim=1, keepdim=True).values.clamp(min=1e-12)  # [K,1]
            importance = importance / q                    # each row crosses 1.0 at its own (1-sp) point
        scope = 'global'                                   # fairness is meaningful only for a shared budget
    if sens_row is not None:
        s = sens_row.to(device).float.clamp(min=0).view(-1)  # [K] per-output-channel sensitivity
        importance = importance * s.pow(sens_exp).view(-1, 1)  # tilt budget toward high-sensitivity rows
    mask = _threshold_mask(importance, sparsity, scope)
    # OBS compensation (identical to wanda_mask_and_obs / sparse_with_prism)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag
    W_comp = W.clone
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def fisher_sal_mask_and_obs(W, X, sparsity, device, scope='global', fisher_M=None, shrink=1.0):
    """EXACT diagonal empirical-Fisher saliency mask + OBS. NON-Sinkhorn, PRISM-orthogonal.

    The per-weight 2nd-order pruning saliency is ½·F_ij·w_ij², F_ij = E[(∂L/∂w_ij)²] = E[g_i² x_j²]
    (g=∂L/∂y). Wanda uses F_ij≈E[x_j²] (drops g ⇒ output Fisher G=I); the factorized sens_wanda
    uses F_ij≈E[g_i²]E[x_j²] (⇒ s_i reweights only ALLOCATION). fisher_M = the EXACT joint
    E[g_i² x_j²] (compute_fisher_saliency.py) keeps the (g_i²,x_j²) correlation ⇒ it reweights
    COLUMNS DIFFERENTLY PER ROW — the within-row selection lever Wanda lacks. saliency = M_ij·w_ij²;
    prune smallest (global or per_row), then the SAME OBS as every other recipe. Fully non-Sinkhorn,
    one backward pass (≤ PRISM). Falls back to Wanda ‖X‖² if M is missing. Returns (W_comp, mask)."""
    K, N = W.shape
    W = W.float.to(device)
    if sparsity <= 0.0:
        return W.clone, torch.ones(K, N, device=device)
    X = X.float.to(device)
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    if fisher_M is not None:
        M = fisher_M.to(device).float.clamp(min=1e-30)   # [K,N] = E[g_i² x_j²]
        if shrink < 1.0:
            # Robustness knob (the S=256 over-fit lesson): shrink the NOISY within-row INTERACTION
            # toward the robust rank-1 (row×col) log-factorization. λ=1 exact Fisher; λ=0 rank-1
            # (per-row column ordering → row-independent ≈ Wanda within-row); 0<λ<1 partial signal.
            L = M.log
            g = L.mean; rdev = L.mean(1, keepdim=True) - g; cdev = L.mean(0, keepdim=True) - g
            base = g + rdev + cdev                          # additive 2-way model (no interaction)
            M = (base + shrink * (L - base)).exp
    else:                                                   # fallback: Wanda (G=I)
        M = (X * X).sum(0).clamp(min=0).view(1, -1).expand(K, N)
    saliency = M * (W * W)                                  # F_ij · w_ij²  (2nd-order pruning cost)
    mask = _threshold_mask(saliency, sparsity, scope)
    # OBS compensation (identical to every other mask path)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag
    W_comp = W.clone
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def inverse_mu_mask_and_obs(W, X, sparsity, device, n_iter=2, scope='global'):
    """PRISM inverse-μ importance mask + OBS compensation. DIAGNOSTIC ONLY —
    uses Sinkhorn to compute the *mask's* μ1·μ2 factors (req#3 forbids Sinkhorn in
    the deliverable; this exists solely to DECOMPOSE mask-vs-norm). Mirrors
    sparse_with_prism's inverse-importance path (sparse_quant.py:786-834) exactly:
    n_iter iterative refinement of importance = |W|·‖X‖/(μ1μ2), then identical OBS.
    Returns (W_compensated, mask) — the rest of the nosink pipeline is unchanged."""
    K, N = W.shape
    W = W.float.to(device)
    if sparsity <= 0.0:
        return W.clone, torch.ones(K, N, device=device)
    if _sinkhorn_log is None:
        raise RuntimeError("sinkhorn_log unavailable (needed for inverse-μ mask)")
    X = X.float.to(device)
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N], L2 per input channel
    n_prune = int(K * N * sparsity)
    mask = torch.ones(K, N, device=device)
    current_W = W.clone
    for _ in range(n_iter):                                # iterative μ refinement (n=2)
        W_for_sink = current_W.clone
        zero = current_W.abs < 1e-10
        if zero.any:
            W_for_sink[zero] = torch.randn(int(zero.sum.item), device=device) * 1e-8
        _, mu1, mu2 = _sinkhorn_log(W_for_sink, order=16)  # mu1=[N] col, mu2=[K] row
        importance = W.abs * act_norms.view(1, -1) / (mu1.view(1, -1) * mu2.view(-1, 1) + 1e-6)
        mask = _threshold_mask(importance, sparsity, scope)  # global=PRISM; per_row=diagnostic
        current_W = W * mask
    # OBS compensation (identical to wanda_mask_and_obs / sparse_with_prism)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag
    W_comp = W.clone
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def scale_mask_and_obs(W, X, sparsity, device, n_iter=2, scope='global'):
    """NON-Sinkhorn inverse-SCALE importance mask + OBS — the deliverable analog of the
    inverse-μ mask. Replaces PRISM's Sinkhorn μ1·μ2 in the pruning importance with
    CLOSED-FORM per-column/row weight scales (sparse-aware std over survivors):
        importance = |W|·‖X‖ / (col_std · row_std),  top-k (global or per_row), iterated.
    This makes the importance SCALE-INVARIANT across rows/cols (the property M1 shows
    Wanda lacks → ~22% row starvation), WITHOUT any Sinkhorn iteration → strictly cheaper
    calibration than PRISM. OBS is identical to the wanda/inverse_mu paths.
    Returns (W_compensated, mask)."""
    K, N = W.shape
    W = W.float.to(device)
    if sparsity <= 0.0:
        return W.clone, torch.ones(K, N, device=device)
    X = X.float.to(device)
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N], L2 per input channel
    mask = torch.ones(K, N, device=device)
    for _ in range(n_iter):                                # closed-form scale refinement
        col_scale = _masked_std(W, mask, dim=0)            # [N] per-col scale (survivors)
        row_scale = _masked_std(W, mask, dim=1)            # [K] per-row scale (survivors)
        importance = W.abs * act_norms.view(1, -1) / (col_scale.view(1, -1) * row_scale.view(-1, 1))
        mask = _threshold_mask(importance, sparsity, scope)
    # OBS compensation (identical to wanda_mask_and_obs / inverse_mu_mask_and_obs)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag
    W_comp = W.clone
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def dual_mask_and_obs(W, X, sparsity, device, n_iter=2, scope='global'):
    """NON-Sinkhorn inverse-DUAL importance mask + OBS — the WORKING closed-form replica
    of PRISM's inverse-μ mask. Replaces Sinkhorn μ1·μ2 with the DUAL decomposition:
        row_scale = std over row survivors                      (≈ μ2)
        dual_c    = std over col survivors of (W / row_scale)   (≈ μ1, rank-corr 0.99–1.00)
    importance = |W|·‖X‖ / (dual_c · row_scale). The key vs the failed `inverse_scale`:
    dual_c is the ROW-ADJUSTED col std (regresses out row scale the way Sinkhorn's joint
    balancing does), whereas marginal col_std proxies μ1 poorly on q/k (rank 0.55). Fully
    non-Sinkhorn; calibration < PRISM. Under scope='per_row' the /row_scale is a within-row
    no-op → pure /dual_c (needs only the perfect μ1 proxy). Returns (W_compensated, mask)."""
    K, N = W.shape
    W = W.float.to(device)
    if sparsity <= 0.0:
        return W.clone, torch.ones(K, N, device=device)
    X = X.float.to(device)
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N]
    mask = torch.ones(K, N, device=device)
    for _ in range(n_iter):                                # closed-form dual refinement
        row_scale = _masked_std(W, mask, dim=1).clamp(min=1e-8)          # [K] ≈ μ2
        dual_c = _masked_std(W / row_scale.view(-1, 1), mask, dim=0)     # [N] ≈ μ1
        importance = W.abs * act_norms.view(1, -1) / (dual_c.view(1, -1) * row_scale.view(-1, 1))
        mask = _threshold_mask(importance, sparsity, scope)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag
    W_comp = W.clone
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def obs_saliency_mask_and_obs(W, X, sparsity, device, scope='per_row'):
    """NON-Sinkhorn OBS/SparseGPT-style saliency mask + OBS compensation, REUSING the
    same Hessian inverse OBS needs (no extra calibration; strictly < PRISM, no Sinkhorn).
    saliency_ij = W_ij² / [H⁻¹]_jj = the one-shot loss increase from pruning w_ij given
    the survivors compensate (classic OBS). This SUPERSEDES Wanda's importance: Wanda
    uses |W|·‖X‖ = |W|·√H_jj (the Hessian DIAGONAL, ignoring cross-channel correlation);
    [H⁻¹]_jj instead captures column COMPENSABILITY — a channel highly correlated with
    others is cheaper to prune. Default scope='per_row' (each output row keeps 1−sp) so
    NO row-scale proxy is needed — sidestepping cell-E's marginal-row_std failure; the
    per-column [H⁻¹]_jj already handles column fairness. Returns (W_compensated, mask)."""
    K, N = W.shape
    W = W.float.to(device)
    if sparsity <= 0.0:
        return W.clone, torch.ones(K, N, device=device)
    X = X.float.to(device)
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag.clamp(min=1e-12)             # [N] = [H⁻¹]_jj
    saliency = (W ** 2) / H_inv_diag.view(1, -1)           # [K,N] OBS one-shot saliency
    mask = _threshold_mask(saliency, sparsity, scope)
    W_comp = W.clone
    for i in range(K):                                     # OBS with the SAME H_inv
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def _masked_std(W, mask, dim, eps=1e-8):
    """Sparse-aware std along `dim`, over surviving (masked-in) entries only.
    Matches PRISM's sparse-aware normalization (avoids zero-deflated variance)."""
    cnt = mask.sum(dim=dim).clamp(min=1.0)
    Wm = W * mask
    mean = Wm.sum(dim=dim) / cnt
    if dim == 0:
        centered = (W - mean.view(1, -1)) * mask
    else:
        centered = (W - mean.view(-1, 1)) * mask
    var = (centered ** 2).sum(dim=dim) / cnt
    return var.sqrt.clamp(min=eps)


def _robust_scale(s, ratio=10.0):
    """Center a scale vector to unit geometric mean and clamp its ratio to
    [gm/ratio, gm*ratio]. Prevents a near-zero std from amplifying a column into
    fp16 overflow (the NaN we hit) — the closed-form analog of Sinkhorn's log-μ
    clamp. Reconstruction is unaffected by the centering (group-RTN is scale
    invariant per row); the clamp only bounds pathological columns."""
    s = torch.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-8)
    gm = s.log.mean.exp
    return (s / gm).clamp(min=1.0 / ratio, max=ratio)


def compute_norm_scales(W_comp, mask, norm, device, act_abs=None, awq_alpha=0.5):
    """Return (r[K], c[N]) per-row / per-col scales for W_norm = W_comp/(r·c).

    KEY FACT (measured): with per-(row,64-col-group) RTN, a per-ROW scale is
    EXACTLY absorbed by the group scale (same rounded q, same reconstruction) —
    it does nothing. Only the per-COLUMN scale changes the quantization grid.
    So the deliverable norms use r=1 and put all the work in c (the μ1 analog):
      none    : no normalization (c=1)                         [μ1=μ2=1, current base]
      col     : c = sparse-aware per-column weight std          [non-Sinkhorn, closed-form]
      dual    : c = col std of (W/rowstd) then r = row std      [non-Sinkhorn; r is inert]
      sinkhorn: c=mu1, r=mu2 from sinkhorn_log                  [DIAGNOSTIC ONLY — req#3 forbids]
    """
    K, N = W_comp.shape
    ones_r = torch.ones(K, device=device)
    ones_c = torch.ones(N, device=device)
    if norm == "none":
        return ones_r, ones_c
    if norm == "col":
        c = _robust_scale(_masked_std(W_comp, mask, dim=0))     # [N]
        return ones_r, c
    if norm == "dual":
        r = _masked_std(W_comp, mask, dim=1).clamp(min=1e-8)    # [K] (inert; de-biases c)
        W1 = W_comp / r.view(-1, 1)
        c = _robust_scale(_masked_std(W1, mask, dim=0))         # [N]
        return r, c
    if norm == "acol":
        # Activation-aware per-column scale (AWQ/SmoothQuant principle — a genuinely
        # different lever from Sinkhorn's weight-variance balancing). Uses the SAME
        # calibration activations already collected for OBS (no extra forward passes).
        # c = mu_w^(1-a) / mu_x^a  (equivalently 1/awq_scale, since we DIVIDE by c).
        if act_abs is None:
            raise RuntimeError("acol needs activations")
        cnt = mask.sum(dim=0).clamp(min=1.0)
        mu_w = ((W_comp.abs * mask).sum(dim=0) / cnt).clamp(min=1e-8)   # sparse-aware col |W|
        mu_x = act_abs.to(device).float.clamp(min=1e-8)                # [N] mean|X| per channel
        a = float(awq_alpha)
        c = _robust_scale(mu_w.pow(1.0 - a) / mu_x.pow(a))
        return ones_r, c
    if norm == "dacol":
        # dual (row-debiased col-std) blended with activation weighting: c_std * mu_x^-a.
        if act_abs is None:
            raise RuntimeError("dacol needs activations")
        r = _masked_std(W_comp, mask, dim=1).clamp(min=1e-8)
        W1 = W_comp / r.view(-1, 1)
        c_std = _masked_std(W1, mask, dim=0)
        mu_x = act_abs.to(device).float.clamp(min=1e-8)
        a = float(awq_alpha)
        c = _robust_scale(c_std / mu_x.pow(a))
        return r, c
    if norm == "sinkhorn":
        if _sinkhorn_log is None:
            raise RuntimeError("sinkhorn_log unavailable")
        W_sparse = W_comp * mask
        zero = W_sparse.abs < 1e-10
        if zero.any:
            W_sparse = W_sparse.clone
            W_sparse[zero] = torch.randn(int(zero.sum.item), device=device) * 1e-8
        _, mu1, mu2 = _sinkhorn_log(W_sparse, order=16)
        return mu2.to(device).float.view(-1), mu1.to(device).float.view(-1)
    if norm == "sinkhorn_sa":  # DIAGNOSTIC: PRISM's sparse-aware final Sinkhorn (req#3 forbids)
        if _sinkhorn_sa is None:
            raise RuntimeError("sinkhorn_log_sparse_aware unavailable")
        _, mu1, mu2 = _sinkhorn_sa(W_comp * mask, mask, order=16)
        return mu2.to(device).float.view(-1), mu1.to(device).float.view(-1)
    raise ValueError(f"unknown norm {norm}")


def _load_sensitivity(sens_file):
    """Load the cached per-output-channel sensitivity {act_key -> s[K]} (compute_sensitivity.py)."""
    if sens_file is None:
        sens_file = os.path.join(_ROOT, "results", "sensitivity", "sens.pt")
    if not os.path.exists(sens_file):
        raise RuntimeError(f"sensitivity cache not found: {sens_file} (run src/compute_sensitivity.py)")
    return torch.load(sens_file, map_location="cpu")


def allocate_sparsity(sm, model, target_sp, sp_min=0.3, sp_max=0.7):
    """GLOBAL-sensitivity SPARSITY ALLOCATION (the novel mask lever). Given per-matrix global
    loss-sensitivity s_m (= E||dLoss/d(matrix out)||^2, one backward pass; results/sens_alloc/sm_*.pt),
    allocate the per-matrix sparsity sp_m to MINIMIZE the 2nd-order GLOBAL model error
        Delta L ~= (1/2) Sum_m (s_m/K_m) * phi(sp_m),   phi(sp) = -ln(1-sp)  (convex, error grows w/ pruning)
    under a FIXED global budget Sum_m numel_m*sp_m = target_sp*Sum numel_m. This is req#1-compliant
    (targets GLOBAL error, NOT per-matrix error: the flat/no-sensitivity version = per-matrix-error
    min, measured WORSE; the s_m weighting is what makes it global) and non-iterative (closed form +
    a 1-D bisection on the multiplier). Water-fill stationarity: w_m/(1-sp_m)=lam*numel_m =>
    sp_m = clip(1 - (s_m/K_m)/(lam*numel_m), sp_min, sp_max). Returns {act_key -> sp_m}.
    Measured (probe_sens_alloc): recovers 98-99% of the optimal per-matrix water-fill on 3 families."""
    paths = bs.get_layer_paths(model)
    info = {}  # act_key -> (w_m, numel_m)
    for li, layer in enumerate(get_layers(model)):
        for p in paths:
            mod = layer; ok = True
            for part in p.split('.'):
                if not hasattr(mod, part):
                    ok = False; break
                mod = getattr(mod, part)
            if not (ok and isinstance(mod, nn.Linear)):
                continue
            key = f'layer_{li}.{p}'
            K = mod.weight.shape[0]
            s = float(sm.get(key, 0.0))
            if s <= 0:
                continue
            info[key] = (s / max(K, 1), mod.weight.numel)
    if not info:
        return {}
    keys = list(info)
    w = torch.tensor([info[k][0] for k in keys], dtype=torch.float64)
    numel = torch.tensor([float(info[k][1]) for k in keys], dtype=torch.float64)
    budget = target_sp * float(numel.sum)

    def used(lam):
        sp = (1.0 - w / (lam * numel)).clamp(sp_min, sp_max)
        return sp, float((sp * numel).sum)

    lo, hi = 1e-30, 1e30
    for _ in range(200):
        mid = (lo * hi) ** 0.5  # geometric bisection (lam spans many orders of magnitude)
        _, u = used(mid)
        if u < budget:
            lo = mid
        else:
            hi = mid
    sp_final, _ = used(hi)
    return {k: float(sp_final[i]) for i, k in enumerate(keys)}


def _compute_fint_inmemory(model, calibration_data, device, tok_cap=512):
    """In-memory per-matrix joint-Fisher INTERACTION M_ij = E[g_i^2 x_j^2] (aligned g,x), for the
    balanced_fint within-matrix lever. Computed here (NOT cached to disk: the full [K,N] tensors blow the
    per-user disk quota). One backward pass per calibration sequence with forward hooks capturing each
    target Linear's (input x, output-grad g). Accumulators kept on CPU. Returns {act_key: [K,N] cpu}."""
    layer_paths = bs.get_layer_paths(model)
    layers = get_layers(model)
    for _l in layers:
        _l.to(device)
    bs.move_embed_to_device(model, device)
    bs.move_final_layers_to_device(model, device)  # full forward needs final norm + lm_head on device
    targets = {}
    for li, layer in enumerate(layers):
        for p in layer_paths:
            mod = layer; ok = True
            for part in p.split('.'):
                if not hasattr(mod, part):
                    ok = False; break
                mod = getattr(mod, part)
            if ok and isinstance(mod, nn.Linear):
                targets[f'layer_{li}.{p}'] = mod
    caps = {}; cap_in = {}; hooks = []
    def mk(k):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o.retain_grad; caps[k] = o
            cap_in[k] = (inp[0] if isinstance(inp, tuple) else inp).detach
        return h
    for k, mod in targets.items:
        hooks.append(mod.register_forward_hook(mk(k)))
    fint = {k: None for k in targets}
    data = calibration_data
    n_rows = data.shape[0]
    for i in tqdm(range(n_rows), desc="fint"):
        batch = data[i:i + 1].to(device)
        model.zero_grad(set_to_none=True); caps.clear; cap_in.clear
        out = model(batch, labels=batch)
        out.loss.backward
        for k, o in caps.items:
            if o.grad is None:
                continue
            g = o.grad.reshape(-1, o.grad.shape[-1]).float
            x = cap_in[k].reshape(-1, cap_in[k].shape[-1]).float
            m = min(g.shape[0], tok_cap)
            fi = ((g[:m] ** 2).t @ (x[:m] ** 2) / m).cpu
            fint[k] = fi if fint[k] is None else fint[k] + fi
    for hk in hooks:
        hk.remove
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache
    return {k: (v / n_rows) for k, v in fint.items if v is not None}


def apply_wanda_obs_rtn(model, calibration_data, nbits, sparsity_by_type, device='cuda',
                        norm='none', awq_alpha=0.5, mask_mode='wanda', dense_norm='sinkhorn',
                        mask_scope='global', wanda_act_exp=1.0, sens_file=None, sens_exp=0.5,
                        sens_fair=False, fisher_shrink=1.0, col_balance_exp=1.0, balance_per_row=False,
                        group_size=None, no_obs=False, quantizer='rtn', margin_log=None,
                        margin_ctx=None, sparsity_by_matrix=None, rank_combine='sum', rank_magw=1e-3,
                        robust_calib=None, floor_frac=0.0):
    """Quantize every target linear with {Wanda|inverse-μ}-mask + OBS + group-RTN,
    using per-type sparsity. mask_mode='wanda' is the deliverable path (non-Sinkhorn);
    mask_mode='inverse_mu' is DIAGNOSTIC (Sinkhorn μ in the mask only) for the
    mask-vs-norm decomposition.

    dense_norm: normalization for DENSE (sparsity==0) matrices, applied INSTEAD of
    `norm`. Default 'sinkhorn' (standard Sinkhorn) reproduces real PRISM's dense path
    EXACTLY (verified matrix-level: dequant Δ=0) — this holds the dense v_proj (the GQA
    bottleneck) constant + optimal across every cell of the mask-vs-norm 2x2, so the
    only thing that varies is the treatment of the PRUNED matrices. Sinkhorn on a dense
    matrix here is a fair, orthogonal lever (v-dense) given equally to all arms; the
    non-Sinkhorn deliverable can use dense_norm='col' (verified nearly as good: 0.1988
    vs 0.1971 recon). Set dense_norm=None to inherit `norm` (legacy behavior).
    group_size: RTN group size for survivor quantization (None -> module default
    GROUP_SIZE=64). Exposed so the camera-ready audit can MATCH the standard
    baselines' group-128 (equal effective bpw) — the fair-comparison test.

    Returns (model, stats) with stats=(type, numel, sparsity)."""
    gsize = GROUP_SIZE if group_size is None else int(group_size)
    fint_cache = None
    if mask_mode == 'balanced_fint':
        fint_cache = _compute_fint_inmemory(model, calibration_data, device)  # {act_key: [K,N] cpu}
    layer_activations = bs.collect_activations(model, calibration_data, device)
    ptb_acts = None
    if mask_mode == 'balanced_stoch' and robust_calib is not None:
        ptb_acts = bs.collect_activations(model, robust_calib, device)   # held-out X per matrix
    robust_norm = None
    if mask_mode == 'balanced_robust' and robust_calib is not None:
        # DISTRIBUTION-ROBUST mask: per-column min(||X||_wiki, ||X||_ptb). Only the SURVIVOR SELECTION
        # is made robust; OBS/quant still use the primary (wiki) activations. Targets held-out
        # generalization (col-||X|| rank-corr across dists is only ~0.55-0.65; mask overfits calib).
        rob_acts = bs.collect_activations(model, robust_calib, device)
        robust_norm = {}
        for k, xw in layer_activations.items:
            xr = rob_acts.get(k)
            if xr is None:
                continue
            def cn(x):
                x = x.float
                if x.dim == 3:
                    x = x.reshape(-1, x.shape[-1])
                return torch.norm(x[:min(x.shape[0], 256)].to(device), dim=0)
            robust_norm[k] = torch.minimum(cn(xw), cn(xr)).cpu
        del rob_acts
    # collect_activations moves every transformer layer to `device` and never offloads,
    # leaving the WHOLE fp16 model resident on GPU throughout the OBS build (peak ~= model
    # size + H_inv). For big models (gemma-7b, ~17GB) that OOMs on a memory-contended MIG
    # slice. Offload layers back to CPU here; the per-layer `layer.to(device)` in the loop
    # below brings them up one at a time (numerically identical, memory-only change).
    for _l in get_layers(model):
        _l.to("cpu")
    torch.cuda.empty_cache
    layer_paths = bs.get_layer_paths(model)
    _nheads = getattr(getattr(model, 'config', None), 'num_attention_heads', None)
    sens_cache = _load_sensitivity(sens_file) if mask_mode == 'sens_wanda' else None
    fisher_dir = os.path.join(_ROOT, "results", "fisher_sal") if mask_mode == 'fisher_sal' else None
    stats = []
    for layer_idx, layer in enumerate(tqdm(get_layers(model), desc=f"WOR {nbits}b")):
        layer = layer.to(device)
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue
            if not isinstance(linear, nn.Linear):
                continue
            t = type_of(attr_path)
            act_key0 = f'layer_{layer_idx}.{attr_path}'
            if sparsity_by_matrix is not None and act_key0 in sparsity_by_matrix:
                sp = sparsity_by_matrix[act_key0]        # global-sensitivity allocation (per-matrix)
            else:
                sp = sparsity_by_type.get(t, 0.70)
            W = linear.weight.data.clone
            bias = linear.bias.data.clone if linear.bias is not None else None
            act_key = f'layer_{layer_idx}.{attr_path}'
            acts = layer_activations.get(act_key, None)
            act_abs = None
            if acts is None:
                # no activations -> cannot OBS; fall back to dense quant of W
                W_comp, mask = W.float.to(device), torch.ones_like(W, device=device).float
            else:
                acts_d = acts.to(device)
                if mask_mode == 'inverse_mu':
                    W_comp, mask = inverse_mu_mask_and_obs(W, acts_d, sp, device, scope=mask_scope)
                elif mask_mode == 'inverse_scale':
                    W_comp, mask = scale_mask_and_obs(W, acts_d, sp, device, scope=mask_scope)
                elif mask_mode == 'inverse_dual':
                    W_comp, mask = dual_mask_and_obs(W, acts_d, sp, device, scope=mask_scope)
                elif mask_mode == 'obs_saliency':
                    W_comp, mask = obs_saliency_mask_and_obs(W, acts_d, sp, device, scope=mask_scope)
                elif mask_mode == 'sens_wanda':
                    sens_row = sens_cache.get(act_key) if sens_cache is not None else None
                    W_comp, mask = sens_wanda_mask_and_obs(W, acts_d, sp, device, scope=mask_scope,
                                                           sens_row=sens_row, sens_exp=sens_exp,
                                                           fair=sens_fair)
                elif mask_mode == 'balanced':
                    W_comp, mask = balanced_mask_and_obs(W, acts_d, sp, device, col_exp=col_balance_exp,
                                                         per_row_thresh=balance_per_row, no_obs=no_obs)
                elif mask_mode == 'balanced_cond':
                    # NON-SEPARABLE conditioning-modulated balance (#28): the column-balance SUPPRESSION
                    # exponent is modulated PER-COLUMN by uniqueness u_j=1/[H^-1]_jj -- well-conditioned
                    # (unique/spanning) columns get LESS suppression (protected, kept more), redundant
                    # columns MORE. Couples balance STRENGTH to conditioning (wanda has no balance => cannot
                    # express => CoBALT-specific). One-shot, global. gamma=0.5 modulation depth.
                    Wd = W.float.to(device)
                    Xh = acts_d.float
                    if Xh.dim == 3:
                        Xh = Xh.reshape(-1, Xh.shape[-1])
                    Xh = Xh[:min(Xh.shape[0], 256)]
                    Kc, Nc = Wd.shape
                    H_inv = compute_hessian_inverse(Xh, damping=None); H_inv_diag = H_inv.diag
                    u = (1.0 / (H_inv_diag + 1e-12)).clamp(min=0)
                    u = (u / (u.mean + 1e-12))                         # normalized uniqueness [N]
                    b = col_balance_exp; gamma = 0.5
                    beta_j = (b * (1.0 - gamma * (u - 1.0))).clamp(min=0.0, max=2.0 * b).view(1, -1)
                    imp = Wd.abs * torch.norm(Xh, dim=0).view(1, -1)
                    kr, kc = int(Nc * sp), int(Kc * sp)
                    if kr > 0:
                        imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
                    if kc > 0:
                        qc = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)  # [1,N]
                        imp = imp / qc.pow(beta_j)                        # PER-COLUMN exponent
                    mask = _threshold_mask(imp, sp, scope='global')
                    if no_obs:
                        W_comp = Wd * mask
                    else:
                        P = (Wd * (1.0 - mask)) / H_inv_diag.view(1, -1)
                        W_comp = (Wd - P @ H_inv) * mask
                elif mask_mode == 'balanced_perhead':
                    # ORTHOGONAL granularity (#30): for o_proj (input channels head-partitioned) add a
                    # per-HEAD-block balance term (global balance leaves per-head keep-rate CV 0.21-0.24,
                    # MEASURED). Each attention head's o_proj input block competes fairly => proportional
                    # head contribution. Other matrices = standard balanced (no input-head structure).
                    Wd = W.float.to(device)
                    Xh = acts_d.float
                    if Xh.dim == 3:
                        Xh = Xh.reshape(-1, Xh.shape[-1])
                    Xh = Xh[:min(Xh.shape[0], 256)]
                    Kh, Nh = Wd.shape
                    imp = Wd.abs * torch.norm(Xh, dim=0).view(1, -1)
                    kr, kc = int(Nh * sp), int(Kh * sp)
                    if kr > 0:
                        imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
                    if col_balance_exp > 0 and kc > 0:
                        imp = imp / torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30).pow(col_balance_exp)
                    if t == 'o' and _nheads and Nh % _nheads == 0 and (Nh // _nheads) > 1:
                        # per-head-block normalization (soft equalization across heads)
                        hd = Nh // _nheads
                        impb = imp.view(Kh, _nheads, hd)
                        kph = max(int(hd * sp), 1)
                        qh = torch.kthvalue(impb.reshape(Kh * _nheads, hd), kph, dim=1).values.view(Kh, _nheads, 1).clamp(min=1e-30)
                        imp = (impb / qh.pow(col_balance_exp)).view(Kh, Nh)
                    mask = _threshold_mask(imp, sp, scope='global')
                    if no_obs:
                        W_comp = Wd * mask
                    else:
                        H_inv = compute_hessian_inverse(Xh, damping=None); H_inv_diag = H_inv.diag
                        P = (Wd * (1.0 - mask)) / H_inv_diag.view(1, -1)
                        W_comp = (Wd - P @ H_inv) * mask
                elif mask_mode == 'wanda_span':
                    # FAIRNESS CONTROL for balanced_span: apply the SAME spanning/unique-variance column
                    # weight to plain per-row WANDA (NO column balance). If wanda_span improves wanda as
                    # much as balanced_span improves cobalt, spanning is a UNIVERSAL lever (not CoBALT-
                    # specific) => the awclip trap. Isolates spanning from column balance.
                    Wd = W.float.to(device)
                    Xs = acts_d.float
                    if Xs.dim == 3:
                        Xs = Xs.reshape(-1, Xs.shape[-1])
                    Xs = Xs[:min(Xs.shape[0], 256)]
                    H_inv = compute_hessian_inverse(Xs, damping=None); H_inv_diag = H_inv.diag
                    uniq = (1.0 / (H_inv_diag + 1e-12)).clamp(min=0)
                    uniq = (uniq / (uniq.mean + 1e-12)).pow(0.5).view(1, -1)
                    imp = Wd.abs * torch.norm(Xs, dim=0).view(1, -1) * uniq
                    mask = _threshold_mask(imp, sp, scope='per_row')
                    if no_obs:
                        W_comp = Wd * mask
                    else:
                        P = (Wd * (1.0 - mask)) / H_inv_diag.view(1, -1)
                        W_comp = (Wd - P @ H_inv) * mask
                elif mask_mode in ('balanced_spectral', 'balanced_specblend'):
                    # SPECTRAL-preserving mask (#42, NOVEL, categorically distinct: Jaccard 0.30 vs balanced).
                    # Importance = |A_k| = top-k SVD reconstruction of the activation-weighted matrix
                    # A = W*||X|| -> keep entries carrying the top singular SUBSPACE (preserve the layer's
                    # dominant output directions), NOT per-entry magnitude. specblend = geometric blend with
                    # the magnitude importance (guard against discarding individually-salient weights).
                    Wd = W.float.to(device)
                    Xsp = acts_d.float
                    if Xsp.dim == 3:
                        Xsp = Xsp.reshape(-1, Xsp.shape[-1])
                    Xsp = Xsp[:min(Xsp.shape[0], 256)]
                    cnorm = torch.norm(Xsp, dim=0)
                    A = Wd * cnorm.view(1, -1)
                    kk = min(32, min(A.shape) - 1)
                    U, Sv, Vh = torch.svd_lowrank(A, q=kk)
                    Ak = (U * Sv.unsqueeze(0)) @ Vh.t
                    imp_mag = Wd.abs * cnorm.view(1, -1)
                    if mask_mode == 'balanced_specblend':
                        imp = imp_mag.clamp(min=1e-30).pow(0.5) * Ak.abs.clamp(min=1e-30).pow(0.5)
                    else:
                        imp = Ak.abs
                    mask = balanced_keepmask_local(imp, sp, col_balance_exp)
                    if no_obs:
                        W_comp = Wd * mask
                    else:
                        H_inv = compute_hessian_inverse(Xsp, damping=None); H_inv_diag = H_inv.diag
                        P = (Wd * (1.0 - mask)) / H_inv_diag.view(1, -1)
                        W_comp = (Wd - P @ H_inv) * mask
                elif mask_mode == 'balanced_span':
                    # SPANNING/conditioning balance (#25): softer column balance (col_exp, deploy 0.4) with
                    # the column importance up-weighted by UNIQUE variance (1/[H^-1]_jj)^0.5 -> keep a
                    # well-conditioned, SPANNING survivor set (better OBS reconstruction off-distribution).
                    # Screen-top candidate; one-shot, global, non-grid. H_inv reused for the weight+OBS.
                    Wd = W.float.to(device)
                    Xs = acts_d.float
                    if Xs.dim == 3:
                        Xs = Xs.reshape(-1, Xs.shape[-1])
                    Xs = Xs[:min(Xs.shape[0], 256)]
                    H_inv = compute_hessian_inverse(Xs, damping=None); H_inv_diag = H_inv.diag
                    uniq = (1.0 / (H_inv_diag + 1e-12)).clamp(min=0)
                    uniq = (uniq / (uniq.mean + 1e-12)).pow(0.5).view(1, -1)
                    imp = Wd.abs * torch.norm(Xs, dim=0).view(1, -1) * uniq
                    mask = balanced_keepmask_local(imp, sp, col_balance_exp)
                    if no_obs:
                        W_comp = Wd * mask
                    else:
                        P = (Wd * (1.0 - mask)) / H_inv_diag.view(1, -1)
                        W_comp = (Wd - P @ H_inv) * mask
                elif mask_mode == 'balanced_exact':
                    # EXACT doubly-degree-balanced support (#18): HARD column balance (CV ~4x < soft-beta)
                    # retaining magnitude info. Sharpens the collapse-rescue mechanism to its hard form.
                    Wd = W.float.to(device)
                    Xe = acts_d.float
                    if Xe.dim == 3:
                        Xe = Xe.reshape(-1, Xe.shape[-1])
                    Xe = Xe[:min(Xe.shape[0], 256)]
                    imp = Wd.abs * torch.norm(Xe, dim=0).view(1, -1)
                    mask = exact_doubly_balanced_mask(imp, sp)
                    if no_obs:
                        W_comp = Wd * mask
                    else:
                        H_inv = compute_hessian_inverse(Xe, damping=None); H_inv_diag = H_inv.diag
                        P = (Wd * (1.0 - mask)) / H_inv_diag.view(1, -1)
                        W_comp = (Wd - P @ H_inv) * mask
                elif mask_mode in ('balanced_floor', 'wanda_floor'):
                    # HARD column-degree-FLOOR (attempt-11 #21): take the base survivor set (balanced or
                    # plain wanda) and enforce a min per-column survivor count, holding global sparsity
                    # EXACTLY constant. Rescues the OBS-uncompensable DEAD columns the soft mask leaves
                    # (MEASURED 0.1-2% at sp0.7, src/probe_colfloor.py). 'wanda_floor' = the S4 attribution
                    # control (same floor on wanda's per-row importance) to test if the floor is universal.
                    Wd = W.float.to(device)
                    Xf = acts_d.float
                    if Xf.dim == 3:
                        Xf = Xf.reshape(-1, Xf.shape[-1])
                    Xf = Xf[:min(Xf.shape[0], 256)]
                    Kf, Nf = Wd.shape
                    imp = Wd.abs * torch.norm(Xf, dim=0).view(1, -1)   # Wanda base importance
                    if mask_mode == 'balanced_floor':
                        # replicate CoBALT's row+col quantile self-normalization for BOTH the base mask
                        # and the floor ranking, so the rescue respects column balance too.
                        kr, kc = int(Nf * sp), int(Kf * sp)
                        if kr > 0:
                            imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
                        if col_balance_exp > 0 and kc > 0:
                            imp = imp / torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30).pow(col_balance_exp)
                        base = _threshold_mask(imp, sp, scope='global')
                    else:  # wanda_floor: plain per-row wanda base + raw importance ranking
                        base = _threshold_mask(imp, sp, scope='per_row')
                    mask = column_floor_mask(base, imp, sp, floor_frac)
                    if no_obs:
                        W_comp = Wd * mask
                    else:
                        H_inv = compute_hessian_inverse(Xf, damping=None); H_inv_diag = H_inv.diag
                        P = (Wd * (1.0 - mask)) / H_inv_diag.view(1, -1)
                        W_comp = (Wd - P @ H_inv) * mask
                elif mask_mode == 'balanced_rank':
                    # NOVEL balance FORM (doubly-balanced RANK-SUM, one-shot / non-iterative). The deployed
                    # balanced mask soft-normalizes importance by row+col quantiles then a GLOBAL top-k.
                    # This instead scores each entry by its PERCENTILE RANK within its row PLUS within its
                    # column (both in [0,1]); keep the (1-sp) with the highest rank-sum. Keeps entries that
                    # are important BOTH within their row AND their column => intrinsically balanced along
                    # both axes without a global-scale artifact, and it is INVARIANT to per-row/col scale
                    # (so no starvation). Different STRUCTURE, not a new importance signal (which degraded).
                    Wd = W.float.to(device)
                    Xr = acts_d.float
                    if Xr.dim == 3:
                        Xr = Xr.reshape(-1, Xr.shape[-1])
                    Xr = Xr[:min(Xr.shape[0], 256)]
                    imp = Wd.abs * torch.norm(Xr, dim=0).view(1, -1)
                    Kr, Nr = imp.shape
                    # percentile rank within each row and within each column (argsort-argsort / size)
                    rrank = imp.argsort(dim=1).argsort(dim=1).float / max(Nr - 1, 1)   # [K,N] in [0,1]
                    crank = imp.argsort(dim=0).argsort(dim=0).float / max(Kr - 1, 1)
                    b = col_balance_exp
                    if rank_combine == 'min':      # keep entries high-rank in BOTH axes (selective)
                        score = torch.minimum(rrank, b * crank if b > 0 else rrank)
                    elif rank_combine == 'prod':   # geometric-ish: both must be high
                        score = rrank * (crank.clamp(min=1e-6) ** b)
                    elif rank_combine == 'magtie': # rank-sum blended with normalized magnitude (rank_magw
                        # controls the blend: small=pure rank; large=magnitude/base-like). Fixes the qwen
                        # collapse where rank alone discards the |W|.||X|| magnitude info.
                        flat = imp.reshape(-1)
                        imp_pr = (flat.argsort.argsort.float / max(Kr * Nr - 1, 1)).view(Kr, Nr)
                        score = rrank + b * crank + rank_magw * (1.0 + b) * imp_pr
                    else:                          # 'sum' (default)
                        score = rrank + b * crank
                    n_keep = int(Kr * Nr * (1.0 - sp))
                    thr = torch.kthvalue(score.view(-1), max(Kr * Nr - n_keep, 1)).values
                    mask = (score.view(-1) > thr).view(Kr, Nr).float
                    if no_obs:
                        W_comp = Wd * mask
                    else:
                        H_inv = compute_hessian_inverse(Xr, damping=None); H_inv_diag = H_inv.diag
                        P = (Wd * (1.0 - mask)) / H_inv_diag.view(1, -1)
                        W_comp = (Wd - P @ H_inv) * mask
                elif mask_mode == 'balanced_stoch':
                    # STOCHASTIC HELD-OUT selection (#14): generate K importance-perturbed balanced masks,
                    # keep the one with lowest HELD-OUT (ptb) reconstruction error. Selects the survivor set
                    # that GENERALIZES best (not the calib-certificate min, which mispredicts downstream).
                    Wd = W.float.to(device)
                    Xw = acts_d.float
                    if Xw.dim == 3:
                        Xw = Xw.reshape(-1, Xw.shape[-1])
                    Xw = Xw[:min(Xw.shape[0], 256)]
                    Xp = ptb_acts.get(act_key) if ptb_acts is not None else None
                    imp0 = Wd.abs * torch.norm(Xw, dim=0).view(1, -1)
                    H_inv = compute_hessian_inverse(Xw, damping=None); H_inv_diag = H_inv.diag
                    if Xp is not None:
                        Xp = Xp.float.to(device)
                        if Xp.dim == 3:
                            Xp = Xp.reshape(-1, Xp.shape[-1])
                        Xp = Xp[:min(Xp.shape[0], 256)]
                    best = None; best_err = float('inf')
                    K_cand = 5
                    for kk in range(K_cand):
                        if kk == 0:
                            impk = imp0
                        else:
                            g = torch.randn_like(imp0) * (0.3 + 0.1 * kk)
                            impk = imp0 * torch.exp(g)
                        mk = balanced_keepmask_local(impk, sp, col_balance_exp)
                        P = (Wd * (1.0 - mk)) / H_inv_diag.view(1, -1)
                        Wc = (Wd - P @ H_inv) * mk
                        D = Wc - Wd
                        if Xp is not None:
                            err = float((Xp @ D.t).pow(2).sum)   # held-out ||X_ptb D^T||^2
                        else:
                            err = float((Xw @ D.t).pow(2).sum)
                        if err < best_err:
                            best_err, best, best_mk = err, Wc, mk
                    W_comp, mask = (best if not no_obs else Wd * best_mk), best_mk
                elif mask_mode == 'balanced_decorr':
                    # REDUNDANCY/DIVERSITY lever (#10/#12): discount input channels whose activations are
                    # REDUNDANT (predictable from others) so the survivor set is more DECORRELATED/spanning.
                    # redundancy_j = mean_k |corr(X_j, X_k)| (from the activation Gram); keep low-redundancy
                    # (independent) channels. importance = |W|.||X||.(1 - redundancy). One-shot, non-iterative.
                    Wd = W.float.to(device)
                    Xc = acts_d.float
                    if Xc.dim == 3:
                        Xc = Xc.reshape(-1, Xc.shape[-1])
                    Xc = Xc[:min(Xc.shape[0], 256)]
                    G = Xc.t @ Xc
                    d = G.diagonal.clamp(min=1e-12).sqrt
                    corr = (G / d.view(-1, 1) / d.view(1, -1)).abs
                    redund = (corr.mean(1) - 1.0 / corr.shape[0]).clamp(0, 1)   # exclude self
                    xn = torch.norm(Xc, dim=0)
                    imp_base = Wd.abs * xn.view(1, -1) * (1.0 - redund).clamp(min=1e-3).view(1, -1)
                    W_comp, mask = balanced_mask_and_obs(W, acts_d, sp, device, col_exp=col_balance_exp,
                                                         no_obs=no_obs, imp_base=imp_base)
                elif mask_mode == 'balanced_outlier':
                    # NON-2nd-order STRUCTURAL lever: LLMs have massive-activation (outlier) channels the
                    # column-BALANCE actively down-weights (balance pushes survivors AWAY from high-energy
                    # cols). Protect the top outlier channels (by max|X|/median peakiness) from the column
                    # balancing so their critical pathways survive, balance the rest. col_balance_exp still
                    # sets balance strength; rank_magw REUSED as the outlier fraction (top-f cols exempt).
                    Wd = W.float.to(device)
                    Xo = acts_d.float
                    if Xo.dim == 3:
                        Xo = Xo.reshape(-1, Xo.shape[-1])
                    Xo = Xo[:min(Xo.shape[0], 256)]
                    xn = torch.norm(Xo, dim=0)
                    pe0 = Xo.abs.amax(0) / Xo.abs.median(0).values.clamp(min=1e-8)  # per-col peakiness
                    f = rank_magw if 0 < rank_magw < 0.5 else 0.02
                    thr_o = torch.quantile(pe0, 1.0 - f)
                    boost = torch.where(pe0 >= thr_o, torch.full_like(xn, 4.0), torch.ones_like(xn))
                    imp_base = Wd.abs * xn.view(1, -1) * boost.view(1, -1)   # outlier cols boosted 4x
                    W_comp, mask = balanced_mask_and_obs(W, acts_d, sp, device, col_exp=col_balance_exp,
                                                         no_obs=no_obs, imp_base=imp_base)
                elif mask_mode == 'balanced_robust':
                    Wd = W.float.to(device)
                    rn = robust_norm.get(act_key) if robust_norm is not None else None
                    imp_base = (Wd.abs * rn.to(device).view(1, -1)) if rn is not None else None
                    W_comp, mask = balanced_mask_and_obs(W, acts_d, sp, device, col_exp=col_balance_exp,
                                                         no_obs=no_obs, imp_base=imp_base)
                elif mask_mode == 'balanced_obs':
                    # OBS-OPTIMAL saliency into the balanced thresholding: the provably-correct pruning cost
                    # UNDER OBS compensation is w_ij^2 / [H^-1]_jj (classic OBS), vs Wanda's |W|.||X|| which
                    # ignores cross-channel compensability. Feed |W|.sqrt(1/[H^-1]_jj) as the balanced
                    # importance (sqrt so it composes with the |W| like the energy form). Closes the
                    # 'which importance signal' axis with its theoretically-optimal member.
                    Wd = W.float.to(device)
                    Xo = acts_d.float
                    if Xo.dim == 3:
                        Xo = Xo.reshape(-1, Xo.shape[-1])
                    Xo = Xo[:min(Xo.shape[0], 256)]
                    Hinv0 = compute_hessian_inverse(Xo, damping=None)
                    hjj = Hinv0.diag.clamp(min=1e-12)                 # [N] = [H^-1]_jj
                    imp_base = Wd.abs * (1.0 / hjj).sqrt.view(1, -1)
                    W_comp, mask = balanced_mask_and_obs(W, acts_d, sp, device, col_exp=col_balance_exp,
                                                         no_obs=no_obs, imp_base=imp_base)
                elif mask_mode == 'balanced_lev':
                    # SET-GEOMETRY / CONDITIONING lever (non-saliency, non-Fisher): weight the balanced
                    # importance by each input channel's RIDGE-LEVERAGE lev_j = [G(G+lam I)^-1]_jj, G=X^T X.
                    # High leverage = independent/unique channel = HARD to OBS-compensate if pruned => keep.
                    # Low leverage = redundant channel = cheaply compensated => prunable. One-shot (one solve),
                    # non-iterative. Replaces Wanda's marginal ||X|| energy with a conditioning signal.
                    Wd = W.float.to(device)
                    Xl = acts_d.float
                    if Xl.dim == 3:
                        Xl = Xl.reshape(-1, Xl.shape[-1])
                    Xl = Xl[:min(Xl.shape[0], 256)]
                    G = Xl.t @ Xl
                    Nn = G.shape[0]
                    lam = 0.01 * (G.diagonal.mean.clamp(min=1e-8))
                    Ginv = torch.linalg.inv(G + lam * torch.eye(Nn, device=device))
                    lev = (G * Ginv.t).sum(1).clamp(min=0)          # diag(G Ginv) [N], in [0,1]
                    imp_base = Wd.abs * lev.view(1, -1).sqrt
                    W_comp, mask = balanced_mask_and_obs(W, acts_d, sp, device, col_exp=col_balance_exp,
                                                         no_obs=no_obs, imp_base=imp_base)
                elif mask_mode == 'balanced_fint':
                    # interaction importance |W|.sqrt(E[g^2 x^2]) fed into the balanced thresholding
                    Wd = W.float.to(device)
                    M = fint_cache.get(act_key) if fint_cache is not None else None
                    imp_base = Wd.abs * M.to(device).float.clamp(min=0).sqrt if M is not None else None
                    W_comp, mask = balanced_mask_and_obs(W, acts_d, sp, device, col_exp=col_balance_exp,
                                                         no_obs=no_obs, imp_base=imp_base)
                elif mask_mode == 'grid_balanced':
                    W_comp, mask = grid_balanced_mask_and_obs(W, acts_d, sp, device,
                                                              col_exp=col_balance_exp, gsize=gsize,
                                                              no_obs=no_obs)
                elif mask_mode == 'fisher_sal':
                    fm_path = os.path.join(fisher_dir, act_key + ".pt") if fisher_dir else None
                    fisher_M = torch.load(fm_path, map_location="cpu") if (fm_path and os.path.exists(fm_path)) else None
                    W_comp, mask = fisher_sal_mask_and_obs(W, acts_d, sp, device, scope=mask_scope,
                                                           fisher_M=fisher_M, shrink=fisher_shrink)
                else:
                    W_comp, mask = wanda_mask_and_obs(W, acts_d, sp, device, scope=mask_scope,
                                                      act_exp=wanda_act_exp)
                Xa = acts_d.float
                if Xa.dim == 3:
                    Xa = Xa.reshape(-1, Xa.shape[-1])
                act_abs = Xa[:min(Xa.shape[0], 256)].abs.mean(0)   # [N] mean|X| per channel
            K, N = W_comp.shape
            # Per-col (μ1 analog) + per-row scaling, then group-RTN on normalized W.
            # Dense matrices (sp==0, e.g. v-dense v_proj) use dense_norm so they are
            # held constant across cells and match real PRISM's dense path exactly.
            eff_norm = dense_norm if (dense_norm is not None and sp <= 0.0) else norm
            r, c = compute_norm_scales(W_comp, mask, eff_norm, device, act_abs=act_abs, awq_alpha=awq_alpha)
            W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
            q, scales, zeros, _ = quantize_rtn(W_norm, [0, 2 ** nbits - 1], group_size=gsize)
            # Fold per-row r into the (already-stored) group scales — no extra overhead.
            if scales.dim == 3:
                scales = scales * r.view(-1, 1, 1)
            else:
                scales = scales * r.view(-1, 1)
            # fp16-storage safety: keep scales representable in half precision.
            scales = torch.nan_to_num(scales, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
            q = q * mask.to(q.dtype)                        # apply mask post-quant (PRISM convention)
            scale2 = c                                       # μ1 analog (per-column), fp16 → ~16/K bpw
            if quantizer != 'rtn' and sp > 0.0 and acts is not None:
                # math4 arms: same mask/OBS/norm, only the survivor quantization
                # differs, at strictly equal total bits (bitmap included).
                import eout_quant as eq
                W_best, info = eq.eout_requantize(W_comp.float, mask, Xa, nbits, gsize,
                                                  r, c, q, scales.float, zeros.float,
                                                  arm=quantizer)
                lin = torch.nn.Linear(N, K, bias=bias is not None)
                lin.weight.data = W_best.half
                if bias is not None:
                    lin.bias.data = bias.half
                lin = lin.to(device)
                setattr(parent, parts[-1], lin)
                if margin_log is not None:
                    ctx = margin_ctx or {}
                    e0 = max(info["e_rtn"], 1e-30)
                    eq.log_margin(margin_log, {
                        "model": ctx.get("model", "?"), "method": quantizer,
                        "sparsity": sp, "bits": nbits,
                        "layer": layer_idx, "type": t,
                        "bprime": info["bprime"], "k": info["k"],
                        "e_rtn": info["e_rtn"], "e_repack": info.get("e_repack"),
                        "e_arm": info["e_arm"], "picked": info.get("picked"),
                        "rel_vs_rtn": info["e_arm"] / e0})
                stats.append((t, W.numel, sp))
                del W, linear, W_comp, mask, q, scales, zeros, W_best
                if acts is not None:
                    del acts
                torch.cuda.empty_cache
                continue
            meta = {'sparsity': sp, 'nbits': nbits, 'requested_nbits': nbits,
                    'group_size': gsize, 'shape': (K, N), 'method': f'wanda_obs_rtn_{norm}'}
            new_layer = bs.SparseQuantLinear(q.half, scales.half, zeros.half,
                                             mask.half, scale2.half, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)
            stats.append((t, W.numel, sp))
            del W, linear, W_comp, mask, q, scales, zeros
            if acts is not None:
                del acts
            torch.cuda.empty_cache
        set_layer(model, layer_idx, layer)
        gc.collect
        torch.cuda.empty_cache
    return model, stats


# --- transformer-layer accessors (mirror benchmark_suite) ---
def get_layers(model):
    return bs.get_transformer_layers(model)


def set_layer(model, idx, layer):
    return bs.set_transformer_layer(model, idx, layer)


def param_fractions(model):
    """Per-type parameter fraction over target linears (for honest global sparsity)."""
    paths = bs.get_layer_paths(model)
    counts = {t: 0 for t in TYPES}
    for layer in get_layers(model):
        for attr_path in paths:
            parts = attr_path.split('.')
            m = layer
            ok = True
            for p in parts:
                if not hasattr(m, p):
                    ok = False
                    break
                m = getattr(m, p)
            if ok and isinstance(m, nn.Linear):
                counts[type_of(attr_path)] += m.weight.numel
    tot = sum(counts.values)
    return counts, tot


def global_sparsity(stats):
    tot = sum(n for _, n, _ in stats)
    pruned = sum(n * sp for _, n, sp in stats)
    return pruned / tot if tot else 0.0


def main:
    global MODEL
    ap = argparse.ArgumentParser
    ap.add_argument("--mode", default="uniform", choices=["uniform", "vdense"])
    ap.add_argument("--norm", default="none",
                    choices=["none", "col", "dual", "acol", "dacol", "sinkhorn", "sinkhorn_sa"],
                    help="per-col(/row) normalization before group-RTN. col/dual=weight-std; "
                         "acol/dacol=activation-aware (AWQ-style); 'sinkhorn*' are DIAGNOSTIC "
                         "upper bounds only (req#3 forbids them in the deliverable).")
    ap.add_argument("--mask", default="wanda",
                    choices=["wanda", "inverse_mu", "inverse_scale", "inverse_dual", "obs_saliency",
                             "sens_wanda", "fisher_sal", "balanced", "balanced_fint", "balanced_lev",
                             "balanced_rank", "balanced_obs", "balanced_robust", "balanced_outlier",
                             "balanced_decorr", "balanced_stoch", "grid_balanced"],
                    help="pruning-mask importance. wanda=|W|·‖X‖ (non-Sinkhorn baseline); "
                         "inverse_mu=PRISM |W|·‖X‖/(μ1μ2) iterative (DIAGNOSTIC: Sinkhorn in the "
                         "mask); inverse_scale=|W|·‖X‖/(col_std·row_std) iterative (NON-Sinkhorn); "
                         "obs_saliency=W²/[H⁻¹]_jj (NON-Sinkhorn OBS/SparseGPT saliency, reuses the "
                         "OBS Hessian; pair with --mask-scope per_row).")
    ap.add_argument("--sens-exp", type=float, default=0.5,
                    help="output-Fisher sensitivity exponent α in the sens_wanda mask importance "
                         "|W|·‖X‖·s_i^α. α=0.5 ⇔ theory-exact (s_i weights the squared saliency); "
                         "α=0 ⇔ plain Wanda; larger α = stronger sensitivity-driven row allocation.")
    ap.add_argument("--sens-file", default=None,
                    help="path to the cached sensitivity dict (default results/sensitivity/sens.pt).")
    ap.add_argument("--col-balance-exp", type=float, default=1.0,
                    help="column-balance strength β for --mask balanced. 0=per-row Wanda (rows only); "
                         "1=full column self-normalization (the measured downstream lever: inverse-μ "
                         "keeps a column-balanced survivor set; saliency masks starve columns).")
    ap.add_argument("--balance-per-row", action="store_true",
                    help="UNIMPEACHABLY non-Sinkhorn variant of --mask balanced: single per-column "
                         "reweighting (AWQ-style, /col-quantile^β) + PER-ROW Wanda threshold (no row "
                         "normalization pass, no global budget, no row/col alternation).")
    ap.add_argument("--fisher-shrink", type=float, default=1.0,
                    help="fisher_sal robustness: shrink the within-row Fisher INTERACTION toward the "
                         "rank-1 (Wanda-like) log-factorization. 1=exact E[g²x²]; 0=rank-1 (≈Wanda "
                         "within-row); 0<λ<1 trades the S=256-overfit risk against the new signal.")
    ap.add_argument("--sens-fair", action="store_true",
                    help="row-fair sens_wanda: normalize each row's saliency by its (1-sp) quantile "
                         "before the s_i^α tilt + global top-k. sens_exp=0 ⇒ per-row Wanda (no scale "
                         "artifact); sens_exp>0 ⇒ pure loss-driven cross-row reallocation.")
    ap.add_argument("--wanda-act-exp", type=float, default=1.0,
                    help="activation-norm exponent γ in the wanda mask importance |W|·‖X‖^γ. "
                         "γ>1 approximates inverse-μ's effective ‖X‖-boost (μ1 anti-corr ‖X‖).")
    ap.add_argument("--mask-scope", default="global", choices=["global", "per_row"],
                    help="threshold scope for wanda/inverse_scale masks. global=single top-k "
                         "(PRISM convention); per_row=each output row keeps exactly (1-sp) "
                         "(guarantees no dead row — the beat-PRISM hypothesis from M1).")
    ap.add_argument("--awq-alpha", type=float, default=0.5,
                    help="activation exponent for acol/dacol (SmoothQuant/AWQ). 0=weight-only.")
    ap.add_argument("--dense-norm", default="sinkhorn",
                    choices=["sinkhorn", "col", "none", "dual", "acol", "dacol", "sinkhorn_sa", "inherit"],
                    help="normalization for DENSE (sp==0) matrices, applied instead of --norm. "
                         "Default 'sinkhorn' reproduces real PRISM's dense v_proj EXACTLY (holds the "
                         "GQA bottleneck constant across the mask-vs-norm 2x2). 'inherit' = use --norm.")
    ap.add_argument("--hold-global", action="store_true",
                    help="in vdense, raise other-type sparsity so GLOBAL stays 0.70")
    ap.add_argument("--target-global", type=float, default=0.70)
    ap.add_argument("--ntest", type=int, default=20)
    ap.add_argument("--downstream", action="store_true",
                    help="after quant, run the SAME downstream suite as the benchmark_suite "
                         "baselines (identical adapters/splits/shots) on the in-memory model.")
    ap.add_argument("--downstream-tasks", default="hellaswag,arc_easy,arc_challenge,mmlu",
                    help="comma-separated task subset, or 'all'.")
    ap.add_argument("--downstream-limit", type=int, default=None,
                    help="subsample N per task (deterministic first-N); None = full split.")
    ap.add_argument("--downstream-csv-dir", default=None,
                    help="dir for per-task downstream CSVs (shared grid). "
                         "Default: results/downstream_grid.")
    ap.add_argument("--technique-tag", default="valor",
                    help="technique label recorded in the downstream CSV 'technique' column.")
    ap.add_argument("--model", default=MODEL, help="model key in bs.MODELS (default gemma-2b).")
    ap.add_argument("--sens-alloc", action="store_true",
                    help="GLOBAL-sensitivity SPARSITY ALLOCATION mask: distribute the fixed global "
                         "sparsity budget across matrices by per-matrix loss-sensitivity s_m "
                         "(results/sens_alloc/sm_<model>.pt), keeping high-s_m matrices denser. "
                         "Non-iterative closed-form water-fill; targets GLOBAL error (req#1). Combines "
                         "with any --mask (default balanced=CoBALT).")
    ap.add_argument("--sens-alloc-file", default=None,
                    help="path to the per-matrix s_m dict (default results/sens_alloc/sm_<model>.pt).")
    ap.add_argument("--rank-combine", default="sum", choices=["sum", "min", "prod", "magtie"],
                    help="balanced_rank score geometry: sum(row+beta*col) | min | prod | magtie.")
    ap.add_argument("--rank-magw", type=float, default=1e-3,
                    help="magtie magnitude-blend weight: 0=pure doubly-rank; large=magnitude/base-like.")
    ap.add_argument("--sp-min", type=float, default=0.3, help="min per-matrix sparsity in the allocation.")
    ap.add_argument("--sp-max", type=float, default=0.7, help="max per-matrix sparsity in the allocation.")
    args = ap.parse_args

    MODEL = args.model
    device = "cuda"
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["seq_len"], n_samples=args.ntest, dataset_key="wikitext2")
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")

    torch.manual_seed(0)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)

    # Build per-type sparsity plan
    if args.mode == "uniform":
        sp_by_type = {t: args.target_global for t in TYPES}
    else:  # vdense
        counts, tot = param_fractions(model)
        v_frac = counts["v"] / tot
        other_sp = args.target_global / (1.0 - v_frac) if args.hold_global else args.target_global
        sp_by_type = {t: other_sp for t in TYPES}
        sp_by_type["v"] = 0.0
        print(f"[plan] v_frac={v_frac:.5f} other_sparsity={other_sp:.5f} "
              f"(hold_global={args.hold_global})", flush=True)

    robust_calib = None
    if args.mask in ('balanced_robust', 'balanced_stoch'):
        robust_calib = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                               seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="ptb")
    sp_by_matrix = None
    if args.sens_alloc:
        sm_path = args.sens_alloc_file or os.path.join(_ROOT, "results", "sens_alloc", f"sm_{MODEL}.pt")
        if not os.path.exists(sm_path):
            raise RuntimeError(f"sens_alloc: {sm_path} not found (run src/compute_loss_sens.py --models {MODEL})")
        sm = torch.load(sm_path, map_location="cpu")
        sp_by_matrix = allocate_sparsity(sm, model, args.target_global, sp_min=args.sp_min, sp_max=args.sp_max)
        vals = list(sp_by_matrix.values)
        print(f"[sens_alloc] {len(vals)} matrices sp in [{min(vals):.3f},{max(vals):.3f}] "
              f"mean={sum(vals)/len(vals):.4f} (target {args.target_global})", flush=True)

    dense_norm = None if args.dense_norm == "inherit" else args.dense_norm
    model, stats = apply_wanda_obs_rtn(model, cal, NBITS, sp_by_type, device,
                                       norm=args.norm, awq_alpha=args.awq_alpha,
                                       mask_mode=args.mask, dense_norm=dense_norm,
                                       mask_scope=args.mask_scope, wanda_act_exp=args.wanda_act_exp,
                                       sens_file=args.sens_file, sens_exp=args.sens_exp,
                                       sens_fair=args.sens_fair, fisher_shrink=args.fisher_shrink,
                                       col_balance_exp=args.col_balance_exp, balance_per_row=args.balance_per_row,
                                       sparsity_by_matrix=sp_by_matrix, rank_combine=args.rank_combine,
                                       rank_magw=args.rank_magw, robust_calib=robust_calib)
    bs.move_final_layers_to_device(model, device)
    model.eval
    gsp = global_sparsity(stats)
    ppl = bs.evaluate_perplexity(model, test, device)
    print(f"RESULT method=wanda_obs_rtn mask={args.mask} scope={args.mask_scope} "
          f"sens_exp={args.sens_exp} sens_fair={args.sens_fair} fisher_shrink={args.fisher_shrink} "
          f"col_balance_exp={args.col_balance_exp} "
          f"act_exp={args.wanda_act_exp} norm={args.norm} dense_norm={args.dense_norm} "
          f"alpha={args.awq_alpha} mode={args.mode} hold_global={args.hold_global} "
          f"global_sparsity={gsp:.4f} ntest={args.ntest} nbits={NBITS} ppl={ppl:.4f}", flush=True)
    out_dir = os.path.join(_ROOT, "results", "nosink")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "nosink.csv"), "a", newline="") as f:
        csv.writer(f).writerow([args.mode, args.hold_global, f"{gsp:.4f}", args.ntest, NBITS,
                                f"{ppl:.4f}", str(sp_by_type), f"norm={args.norm}",
                                f"alpha={args.awq_alpha}"])
    print(f"APPENDED {out_dir}/nosink.csv", flush=True)

    # Downstream-task suite (identical adapters/splits/shots as the baselines), on the
    # already-quantized in-memory model. Mirrors benchmark_suite.run_benchmark's block.
    if args.downstream:
        try:
            from downstream import run_downstream_suite  # noqa: E402
            model.seqlen = bs.EVAL_CONFIG["seq_len"]  # CRB evals read model.seqlen
            ds_dir = args.downstream_csv_dir or os.path.join(_ROOT, "results", "downstream_grid")
            ds_tasks = ("all" if args.downstream_tasks in (None, "all")
                        else [t.strip for t in args.downstream_tasks.split(",")])
            ds_cfg = {
                "timestamp": None,
                "model": MODEL,
                "model_name": name,
                "technique": args.technique_tag,
                "precision": NBITS,
                "sparsity": round(gsp, 4),
                "dataset": "wikitext2",
            }
            run_downstream_suite(model, tok, device, tasks=ds_tasks,
                                 limit=args.downstream_limit, config=ds_cfg,
                                 results_dir=ds_dir, seqlen=bs.EVAL_CONFIG["seq_len"],
                                 verbose=True)
            print(f"DOWNSTREAM_DONE dir={ds_dir}", flush=True)
        except Exception as e:
            print(f"[downstream] suite failed: {e}", flush=True)


if __name__ == "__main__":
    main
