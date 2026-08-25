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
    if scope == 'per_row':
        k_prune = int(N * sparsity)
        if k_prune <= 0:
            return torch.ones_like(importance)
        thr = torch.kthvalue(importance, k_prune, dim=1, keepdim=True).values  # [K,1]
        return (importance > thr).float()
    n_prune = int(K * N * sparsity)
    flat = importance.view(-1)
    thr = torch.kthvalue(flat, n_prune).values
    return (flat > thr).view(K, N).float()


def wanda_mask_and_obs(W, X, sparsity, device, scope='global', act_exp=1.0):
    """Wanda mask (|W|·‖X‖^act_exp, scope global or per_row) + OBS compensation. NO Sinkhorn.
    act_exp>1 boosts the activation exponent: the μ1-proxy diagnostic shows PRISM's inverse-μ
    μ1 ANTI-correlates with ‖X‖, so importance=|W|·‖X‖/μ1 ≈ |W|·‖X‖^(1+β) — i.e. the Sinkhorn
    mask effectively raises the ‖X‖ exponent. This is the closed-form, non-Sinkhorn analog.
    Returns (W_compensated, mask). Wanda+OBS path mirrors sparse_with_prism, no sinkhorn_log."""
    K, N = W.shape
    W = W.float().to(device)
    if sparsity <= 0.0:
        return W.clone(), torch.ones(K, N, device=device)
    X = X.float().to(device)
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N], L2 per input channel
    importance = W.abs() * act_norms.view(1, -1).pow(act_exp)  # Wanda, activation exponent
    mask = _threshold_mask(importance, sparsity, scope)
    # OBS compensation (Hessian only; Sinkhorn-free)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag()
    W_comp = W.clone()
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def balanced_mask_and_obs(W, X, sparsity, device, col_exp=1.0, row_fair=True, per_row_thresh=False,
                          no_obs=False):
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
    W = W.float().to(device)
    if sparsity <= 0.0:
        return W.clone(), torch.ones(K, N, device=device)
    X = X.float().to(device)
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N]
    imp = W.abs() * act_norms.view(1, -1)                  # Wanda base
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
    H_inv_diag = H_inv.diag()
    W_comp = W.clone()
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


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
    W = W.float().to(device)
    if sparsity <= 0.0:
        return W.clone(), torch.ones(K, N, device=device)
    X = X.float().to(device)
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N], L2 per input channel
    importance = W.abs() * act_norms.view(1, -1)           # Wanda base saliency
    if fair:                                               # row-fair: normalize by per-row (1-sp) quantile
        k_prune = int(N * sparsity)
        if k_prune > 0:
            q = torch.kthvalue(importance, k_prune, dim=1, keepdim=True).values.clamp(min=1e-12)  # [K,1]
            importance = importance / q                    # each row crosses 1.0 at its own (1-sp) point
        scope = 'global'                                   # fairness is meaningful only for a shared budget
    if sens_row is not None:
        s = sens_row.to(device).float().clamp(min=0).view(-1)  # [K] per-output-channel sensitivity
        importance = importance * s.pow(sens_exp).view(-1, 1)  # tilt budget toward high-sensitivity rows
    mask = _threshold_mask(importance, sparsity, scope)
    # OBS compensation (identical to wanda_mask_and_obs / sparse_with_prism)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag()
    W_comp = W.clone()
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
    W = W.float().to(device)
    if sparsity <= 0.0:
        return W.clone(), torch.ones(K, N, device=device)
    X = X.float().to(device)
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    if fisher_M is not None:
        M = fisher_M.to(device).float().clamp(min=1e-30)   # [K,N] = E[g_i² x_j²]
        if shrink < 1.0:
            # Robustness knob (the S=256 over-fit lesson): shrink the NOISY within-row INTERACTION
            # toward the robust rank-1 (row×col) log-factorization. λ=1 exact Fisher; λ=0 rank-1
            # (per-row column ordering → row-independent ≈ Wanda within-row); 0<λ<1 partial signal.
            L = M.log()
            g = L.mean(); rdev = L.mean(1, keepdim=True) - g; cdev = L.mean(0, keepdim=True) - g
            base = g + rdev + cdev                          # additive 2-way model (no interaction)
            M = (base + shrink * (L - base)).exp()
    else:                                                   # fallback: Wanda (G=I)
        M = (X * X).sum(0).clamp(min=0).view(1, -1).expand(K, N)
    saliency = M * (W * W)                                  # F_ij · w_ij²  (2nd-order pruning cost)
    mask = _threshold_mask(saliency, sparsity, scope)
    # OBS compensation (identical to every other mask path)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag()
    W_comp = W.clone()
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
    W = W.float().to(device)
    if sparsity <= 0.0:
        return W.clone(), torch.ones(K, N, device=device)
    if _sinkhorn_log is None:
        raise RuntimeError("sinkhorn_log unavailable (needed for inverse-μ mask)")
    X = X.float().to(device)
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N], L2 per input channel
    n_prune = int(K * N * sparsity)
    mask = torch.ones(K, N, device=device)
    current_W = W.clone()
    for _ in range(n_iter):                                # iterative μ refinement (n=2)
        W_for_sink = current_W.clone()
        zero = current_W.abs() < 1e-10
        if zero.any():
            W_for_sink[zero] = torch.randn(int(zero.sum().item()), device=device) * 1e-8
        _, mu1, mu2 = _sinkhorn_log(W_for_sink, order=16)  # mu1=[N] col, mu2=[K] row
        importance = W.abs() * act_norms.view(1, -1) / (mu1.view(1, -1) * mu2.view(-1, 1) + 1e-6)
        mask = _threshold_mask(importance, sparsity, scope)  # global=PRISM; per_row=diagnostic
        current_W = W * mask
    # OBS compensation (identical to wanda_mask_and_obs / sparse_with_prism)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag()
    W_comp = W.clone()
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
    W = W.float().to(device)
    if sparsity <= 0.0:
        return W.clone(), torch.ones(K, N, device=device)
    X = X.float().to(device)
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N], L2 per input channel
    mask = torch.ones(K, N, device=device)
    for _ in range(n_iter):                                # closed-form scale refinement
        col_scale = _masked_std(W, mask, dim=0)            # [N] per-col scale (survivors)
        row_scale = _masked_std(W, mask, dim=1)            # [K] per-row scale (survivors)
        importance = W.abs() * act_norms.view(1, -1) / (col_scale.view(1, -1) * row_scale.view(-1, 1))
        mask = _threshold_mask(importance, sparsity, scope)
    # OBS compensation (identical to wanda_mask_and_obs / inverse_mu_mask_and_obs)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag()
    W_comp = W.clone()
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
    W = W.float().to(device)
    if sparsity <= 0.0:
        return W.clone(), torch.ones(K, N, device=device)
    X = X.float().to(device)
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    act_norms = torch.norm(X, dim=0)                       # [N]
    mask = torch.ones(K, N, device=device)
    for _ in range(n_iter):                                # closed-form dual refinement
        row_scale = _masked_std(W, mask, dim=1).clamp(min=1e-8)          # [K] ≈ μ2
        dual_c = _masked_std(W / row_scale.view(-1, 1), mask, dim=0)     # [N] ≈ μ1
        importance = W.abs() * act_norms.view(1, -1) / (dual_c.view(1, -1) * row_scale.view(-1, 1))
        mask = _threshold_mask(importance, sparsity, scope)
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag()
    W_comp = W.clone()
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
    W = W.float().to(device)
    if sparsity <= 0.0:
        return W.clone(), torch.ones(K, N, device=device)
    X = X.float().to(device)
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag().clamp(min=1e-12)             # [N] = [H⁻¹]_jj
    saliency = (W ** 2) / H_inv_diag.view(1, -1)           # [K,N] OBS one-shot saliency
    mask = _threshold_mask(saliency, sparsity, scope)
    W_comp = W.clone()
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
    return var.sqrt().clamp(min=eps)


def _robust_scale(s, ratio=10.0):
    """Center a scale vector to unit geometric mean and clamp its ratio to
    [gm/ratio, gm*ratio]. Prevents a near-zero std from amplifying a column into
    fp16 overflow (the NaN we hit) — the closed-form analog of Sinkhorn's log-μ
    clamp. Reconstruction is unaffected by the centering (group-RTN is scale
    invariant per row); the clamp only bounds pathological columns."""
    s = torch.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0).clamp(min=1e-8)
    gm = s.log().mean().exp()
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
        mu_w = ((W_comp.abs() * mask).sum(dim=0) / cnt).clamp(min=1e-8)   # sparse-aware col |W|
        mu_x = act_abs.to(device).float().clamp(min=1e-8)                # [N] mean|X| per channel
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
        mu_x = act_abs.to(device).float().clamp(min=1e-8)
        a = float(awq_alpha)
        c = _robust_scale(c_std / mu_x.pow(a))
        return r, c
    if norm == "sinkhorn":
        if _sinkhorn_log is None:
            raise RuntimeError("sinkhorn_log unavailable")
        W_sparse = W_comp * mask
        zero = W_sparse.abs() < 1e-10
        if zero.any():
            W_sparse = W_sparse.clone()
            W_sparse[zero] = torch.randn(int(zero.sum().item()), device=device) * 1e-8
        _, mu1, mu2 = _sinkhorn_log(W_sparse, order=16)
        return mu2.to(device).float().view(-1), mu1.to(device).float().view(-1)
    if norm == "sinkhorn_sa":  # DIAGNOSTIC: PRISM's sparse-aware final Sinkhorn (req#3 forbids)
        if _sinkhorn_sa is None:
            raise RuntimeError("sinkhorn_log_sparse_aware unavailable")
        _, mu1, mu2 = _sinkhorn_sa(W_comp * mask, mask, order=16)
        return mu2.to(device).float().view(-1), mu1.to(device).float().view(-1)
    raise ValueError(f"unknown norm {norm}")


def _load_sensitivity(sens_file):
    """Load the cached per-output-channel sensitivity {act_key -> s[K]} (compute_sensitivity.py)."""
    if sens_file is None:
        sens_file = os.path.join(_ROOT, "results", "sensitivity", "sens.pt")
    if not os.path.exists(sens_file):
        raise RuntimeError(f"sensitivity cache not found: {sens_file} (run src/compute_sensitivity.py)")
    return torch.load(sens_file, map_location="cpu")


def apply_wanda_obs_rtn(model, calibration_data, nbits, sparsity_by_type, device='cuda',
                        norm='none', awq_alpha=0.5, mask_mode='wanda', dense_norm='sinkhorn',
                        mask_scope='global', wanda_act_exp=1.0, sens_file=None, sens_exp=0.5,
                        sens_fair=False, fisher_shrink=1.0, col_balance_exp=1.0, balance_per_row=False,
                        group_size=None, no_obs=False):
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
    layer_activations = bs.collect_activations(model, calibration_data, device)
    # collect_activations moves every transformer layer to `device` and never offloads,
    # leaving the WHOLE fp16 model resident on GPU throughout the OBS build (peak ~= model
    # size + H_inv). For big models (gemma-7b, ~17GB) that OOMs on a memory-contended MIG
    # slice. Offload layers back to CPU here; the per-layer `layer.to(device)` in the loop
    # below brings them up one at a time (numerically identical, memory-only change).
    for _l in get_layers(model):
        _l.to("cpu")
    torch.cuda.empty_cache()
    layer_paths = bs.get_layer_paths(model)
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
            sp = sparsity_by_type.get(t, 0.70)
            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None
            act_key = f'layer_{layer_idx}.{attr_path}'
            acts = layer_activations.get(act_key, None)
            act_abs = None
            if acts is None:
                # no activations -> cannot OBS; fall back to dense quant of W
                W_comp, mask = W.float().to(device), torch.ones_like(W, device=device).float()
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
                elif mask_mode == 'fisher_sal':
                    fm_path = os.path.join(fisher_dir, act_key + ".pt") if fisher_dir else None
                    fisher_M = torch.load(fm_path, map_location="cpu") if (fm_path and os.path.exists(fm_path)) else None
                    W_comp, mask = fisher_sal_mask_and_obs(W, acts_d, sp, device, scope=mask_scope,
                                                           fisher_M=fisher_M, shrink=fisher_shrink)
                else:
                    W_comp, mask = wanda_mask_and_obs(W, acts_d, sp, device, scope=mask_scope,
                                                      act_exp=wanda_act_exp)
                Xa = acts_d.float()
                if Xa.dim() == 3:
                    Xa = Xa.reshape(-1, Xa.shape[-1])
                act_abs = Xa[:min(Xa.shape[0], 256)].abs().mean(0)   # [N] mean|X| per channel
            K, N = W_comp.shape
            # Per-col (μ1 analog) + per-row scaling, then group-RTN on normalized W.
            # Dense matrices (sp==0, e.g. v-dense v_proj) use dense_norm so they are
            # held constant across cells and match real PRISM's dense path exactly.
            eff_norm = dense_norm if (dense_norm is not None and sp <= 0.0) else norm
            r, c = compute_norm_scales(W_comp, mask, eff_norm, device, act_abs=act_abs, awq_alpha=awq_alpha)
            W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
            q, scales, zeros, _ = quantize_rtn(W_norm, [0, 2 ** nbits - 1], group_size=gsize)
            # Fold per-row r into the (already-stored) group scales — no extra overhead.
            if scales.dim() == 3:
                scales = scales * r.view(-1, 1, 1)
            else:
                scales = scales * r.view(-1, 1)
            # fp16-storage safety: keep scales representable in half precision.
            scales = torch.nan_to_num(scales, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
            q = q * mask.to(q.dtype)                        # apply mask post-quant (PRISM convention)
            scale2 = c                                       # μ1 analog (per-column), fp16 → ~16/K bpw
            meta = {'sparsity': sp, 'nbits': nbits, 'requested_nbits': nbits,
                    'group_size': gsize, 'shape': (K, N), 'method': f'wanda_obs_rtn_{norm}'}
            new_layer = bs.SparseQuantLinear(q.half(), scales.half(), zeros.half(),
                                             mask.half(), scale2.half(), bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)
            stats.append((t, W.numel(), sp))
            del W, linear, W_comp, mask, q, scales, zeros
            if acts is not None:
                del acts
            torch.cuda.empty_cache()
        set_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()
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
                counts[type_of(attr_path)] += m.weight.numel()
    tot = sum(counts.values())
    return counts, tot


def global_sparsity(stats):
    tot = sum(n for _, n, _ in stats)
    pruned = sum(n * sp for _, n, sp in stats)
    return pruned / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="uniform", choices=["uniform", "vdense"])
    ap.add_argument("--norm", default="none",
                    choices=["none", "col", "dual", "acol", "dacol", "sinkhorn", "sinkhorn_sa"],
                    help="per-col(/row) normalization before group-RTN. col/dual=weight-std; "
                         "acol/dacol=activation-aware (AWQ-style); 'sinkhorn*' are DIAGNOSTIC "
                         "upper bounds only (req#3 forbids them in the deliverable).")
    ap.add_argument("--mask", default="wanda",
                    choices=["wanda", "inverse_mu", "inverse_scale", "inverse_dual", "obs_saliency",
                             "sens_wanda", "fisher_sal", "balanced"],
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
    args = ap.parse_args()

    device = "cuda"
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["seq_len"], n_samples=args.ntest, dataset_key="wikitext2")
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")

    torch.manual_seed(0)
    if torch.cuda.is_available():
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

    dense_norm = None if args.dense_norm == "inherit" else args.dense_norm
    model, stats = apply_wanda_obs_rtn(model, cal, NBITS, sp_by_type, device,
                                       norm=args.norm, awq_alpha=args.awq_alpha,
                                       mask_mode=args.mask, dense_norm=dense_norm,
                                       mask_scope=args.mask_scope, wanda_act_exp=args.wanda_act_exp,
                                       sens_file=args.sens_file, sens_exp=args.sens_exp,
                                       sens_fair=args.sens_fair, fisher_shrink=args.fisher_shrink,
                                       col_balance_exp=args.col_balance_exp, balance_per_row=args.balance_per_row)
    bs.move_final_layers_to_device(model, device)
    model.eval()
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
                        else [t.strip() for t in args.downstream_tasks.split(",")])
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
    main()
