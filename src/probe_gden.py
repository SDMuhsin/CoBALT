#!/usr/bin/env python3
"""MECHANISM PROBE for the `gden` survivor encoding (go/no-go before any wiring).

gden = a single GLOBAL non-uniform codebook, fit ONCE to the pooled, column-
normalized CoBALT survivor law and WEIGHTED by output energy (A_g*c_j)^2*||X_j||^2,
applied per-group in the (mu, A) affine frame. Panter-Dite density^{1/3} compander
=> MSE-optimal fixed-rate points, CLOSED FORM, one pass, NO iteration, NO per-layer
fit. Storage per group = (mu, A), same 2 values as RTN; codebook stored once.

This probe answers the ONE question A7 demands before companding may be reopened:
  does the codebook actually REDUCE model OUTPUT error tr(D H D^T) vs deployed RTN,
  where H = X^T X is the calibration Hessian? (A7: certificate/marginal-MSE != output.)
It compares, per real gemma-2b matrix at CoBALT's operating point (sp0.5, beta0.5,
3-bit, group-64):
  e_rtn     : deployed uniform group-RTN (grid over ALL entries incl pruned zeros)
  e_gden    : global density^{1/3} codebook, OUTPUT-ENERGY weighted (the proposal)
  e_gden_uw : same, UNWEIGHTED (density only) -- isolates the activation-weight lever
  e_compand : the KILLED A7 per-tensor power-gamma (fit to plain survivor-MSE)
All decoded to original space (x c, post-mask) exactly like the deployed pipeline.
Codebook is built on a SAMPLE of matrices and evaluated on the same (mechanism-
existence check on calibration output error; held-out downstream is the LATER cell).
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks"))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import benchmark_suite as bs  # noqa: E402
import nosink as ns  # noqa: E402
import eout_quant as eq  # noqa: E402
from sinq.sparse_quant import quantize_rtn  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import argparse as _ap
_P = _ap.ArgumentParser()
_P.add_argument("--model", default="gemma-2b")
_P.add_argument("--alphas", default="1.1,1.05,1.0,0.95,0.9,0.85,0.8")
_A, _ = _P.parse_known_args()
MODEL = _A.model
NBITS = 3
GROUP = 64
SP = 0.5
BETA = 0.5
CAP256 = 256


def group_affine(W_norm, mask, gsize):
    """Per-group survivor (mu, A) affine frame; t=(w-mu)/A in [-1,1] over survivors.
    Returns t [K,N], mu [K,ng,1], A [K,ng,1], grouped(bool). Mirrors compand frame."""
    K, N = W_norm.shape
    grouped = N > gsize and N % gsize == 0
    if grouped:
        Wg = W_norm.view(K, N // gsize, gsize)
        Mg = mask.view(K, N // gsize, gsize).bool()
    else:
        Wg = W_norm.unsqueeze(1)
        Mg = mask.unsqueeze(1).bool()
    cnt = Mg.sum(-1, keepdim=True).clamp(min=1.0)
    mu = torch.where(Mg, Wg, torch.zeros_like(Wg)).sum(-1, keepdim=True) / cnt
    A = torch.where(Mg, (Wg - mu).abs(), torch.zeros_like(Wg)).amax(-1, keepdim=True).clamp(min=1e-8)
    empty = ~Mg.any(-1, keepdim=True)
    mu = torch.where(empty, torch.zeros_like(mu), mu)
    A = torch.where(empty, torch.ones_like(A), A)
    t = ((Wg - mu) / A).clamp(-1.0, 1.0)
    return t, mu, A, Mg, grouped


def bincenter_quantize(W_norm, mask, nbits, gsize, alpha=1.0):
    """Bin-CENTERED uniform grid over the SURVIVOR hull [min,max] per group.
    L=2^b bins (NOT L-1 intervals), reconstruction at bin centers, step scaled by
    alpha (alpha=1 => exact bins spanning [min,max]; alpha<1 => range compression).
    MSE-optimal uniform quantizer for a UNIFORM bounded source (the measured survivor
    law). Same 2 stored values/group as RTN, same dense format. Non-iterative, global
    rule. Returns W_hat_norm [K,N] (survivors coded; non-survivors decode via mask)."""
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
    half = (0.5 * (w_max - w_min)).clamp(min=1e-8) / alpha        # range-scaling DOF
    lo = mid - half
    step = (2.0 * half) / L
    q = torch.floor((Wg - lo) / step)
    q = q.clamp(0, L - 1)
    W_hat = lo + (q + 0.5) * step                                 # bin centers
    return W_hat.reshape(K, N)


def build_codebook(t_samples, weights, L, bins=4096, eps=1e-12):
    """Panter-Dite density^{1/3} compander codebook on weighted samples t in [-1,1].
    Non-iterative: one weighted histogram -> compander g -> invert to L levels."""
    device = t_samples.device
    idx = (((t_samples + 1.0) * 0.5) * bins).clamp(0, bins - 1).long()
    wh = torch.zeros(bins, device=device).scatter_add_(0, idx, weights.to(device))
    f = wh / wh.sum().clamp(min=eps)
    lam = f.clamp(min=0).pow(1.0 / 3.0)
    lam = lam / lam.sum().clamp(min=eps)
    cdf = torch.cumsum(lam, 0)                      # compander g(t) in [0,1], length bins
    centers = torch.linspace(-1.0 + 1.0 / bins, 1.0 - 1.0 / bins, bins, device=device)
    targets = (torch.arange(L, device=device) + 0.5) / L
    # invert cdf (monotone nondecreasing) at targets via searchsorted
    pos = torch.searchsorted(cdf.contiguous(), targets.contiguous()).clamp(0, bins - 1)
    codebook = centers[pos]
    codebook, _ = torch.sort(codebook)
    return codebook                                # [L]


def apply_codebook(t, mu, A, Mg, grouped, codebook, K, N):
    """Nearest-codebook quantize t, decode to normalized-W space W_hat_norm [K,N]."""
    d = (t.unsqueeze(-1) - codebook.view(1, 1, 1, -1)).abs()  # [...,L]
    q = d.argmin(-1)
    t_hat = codebook[q]
    W_hat = mu + A * t_hat
    return W_hat.reshape(K, N)


def main():
    device = "cuda"
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    acts_all = bs.collect_activations(model, cal, device)
    for _l in ns.get_layers(model):
        _l.to("cpu")
    torch.cuda.empty_cache()

    layer_paths = bs.get_layer_paths(model)
    n_layers = len(ns.get_layers(model))
    sample_layers = sorted(set([0, n_layers // 2, n_layers - 1]))
    print(f"[probe] {MODEL} layers={n_layers} sample_layers={sample_layers} "
          f"sp={SP} beta={BETA} nbits={NBITS} group={GROUP}", flush=True)

    # ---- PASS 1: build CoBALT objects per sampled matrix, cache them + pooled samples
    cache = []          # (key, W_comp, mask, c, W_norm, H, Xa)
    pooled_t, pooled_w, pooled_wuw = [], [], []
    layers = ns.get_layers(model)
    for li in sample_layers:
        layer = layers[li].to(device)
        for ap in layer_paths:
            parts = ap.split('.')
            parent = layer
            ok = True
            for p in parts[:-1]:
                if not hasattr(parent, p):
                    ok = False
                    break
                parent = getattr(parent, p)
            if not ok or not hasattr(parent, parts[-1]):
                continue
            lin = getattr(parent, parts[-1])
            if not isinstance(lin, torch.nn.Linear):
                continue
            key = f'layer_{li}.{ap}'
            acts = acts_all.get(key)
            if acts is None:
                continue
            W = lin.weight.data.clone()
            W_comp, mask = ns.balanced_mask_and_obs(W, acts.to(device), SP, device, col_exp=BETA)
            r, c = ns.compute_norm_scales(W_comp, mask, 'col', device)
            W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
            Xa = acts.to(device).float()
            if Xa.dim() == 3:
                Xa = Xa.reshape(-1, Xa.shape[-1])
            Xa = Xa[:min(Xa.shape[0], CAP256)]
            H = Xa.t() @ Xa
            colE = (Xa * Xa).sum(0).clamp(min=0)            # ||X_j||^2  [N]
            K, N = W_norm.shape
            t, mu, A, Mg, grouped = group_affine(W_norm, mask, GROUP)
            # output-energy weight per survivor: (A_g * c_j)^2 * ||X_j||^2
            if grouped:
                A_full = A.expand(K, N // GROUP, GROUP).reshape(K, N)
            else:
                A_full = A.expand(K, N)
            step = (A_full * c.view(1, -1))
            wgt = (step * step) * colE.view(1, -1)          # [K,N]
            tflat = t.reshape(K, N)
            m = mask.bool()
            pooled_t.append(tflat[m].detach())
            pooled_w.append(wgt[m].detach())
            pooled_wuw.append(torch.ones_like(wgt[m]).detach())
            cache.append((key, W_comp, mask, c, W_norm, H, r))
        layers[li] = layer.to("cpu")
        torch.cuda.empty_cache()

    T = torch.cat(pooled_t)
    Wt = torch.cat(pooled_w)
    Wuw = torch.cat(pooled_wuw)
    # distribution shape diagnostics (is the survivor law non-uniform at all?)
    def kurt(x):
        x = x.float(); m = x.mean(); s = x.std().clamp(min=1e-12)
        return float(((x - m) / s).pow(4).mean())
    Tabs = T.abs().float()
    sub = Tabs if Tabs.numel() <= 8_000_000 else Tabs[torch.randperm(Tabs.numel(), device=Tabs.device)[:8_000_000]]
    print(f"[shape] pooled survivors n={T.numel()} | t: std={T.std():.4f} "
          f"kurtosis={kurt(T):.3f} (uniform=1.8, gaussian=3.0) "
          f"p50|t|={sub.median():.4f} p99|t|={sub.quantile(0.99):.4f}", flush=True)

    L = 2 ** NBITS
    cb_w = build_codebook(T, Wt, L)
    cb_uw = build_codebook(T, Wuw, L)
    print(f"[codebook] weighted   = {[round(x,3) for x in cb_w.tolist()]}", flush=True)
    print(f"[codebook] unweighted = {[round(x,3) for x in cb_uw.tolist()]}", flush=True)

    # ---- PASS 2: evaluate output error per matrix for each scheme
    ALPHAS = [float(x) for x in _A.alphas.split(",")]     # bin-center range-scaling grid (global DOF)
    tot = {"rtn": 0.0, "gden": 0.0, "gden_uw": 0.0, "compand": 0.0, "saff": 0.0, "smdz": 0.0}
    tot.update({f"binc{a}": 0.0 for a in ALPHAS})
    wins = {"gden": 0, "gden_uw": 0, "compand": 0, "saff": 0, "smdz": 0}
    wins.update({f"binc{a}": 0 for a in ALPHAS})
    nmat = 0
    print(f"\n{'matrix':<34}{'e_rtn':>12}{'gden/rtn':>11}{'compand/rtn':>12}{'saff/rtn':>10}{'binc1.0/rtn':>12}", flush=True)
    for (key, W_comp, mask, c, W_norm, H, r) in cache:
        K, N = W_norm.shape
        m = mask.float()
        # deployed RTN
        q, scales, zeros, _ = quantize_rtn(W_norm, [0, 2 ** NBITS - 1], group_size=GROUP)
        if scales.dim() == 3:
            scales = scales * r.view(-1, 1, 1)
        else:
            scales = scales * r.view(-1, 1)
        scales = torch.nan_to_num(scales, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
        W_rtn = eq.dequant_deployed(q, scales, zeros, m, c)
        e_rtn = eq.eout_sq(W_rtn, W_comp, H)
        # gden (weighted + unweighted): decode in normalized space then x(r*c), post-mask
        t, mu, A, Mg, grouped = group_affine(W_norm, mask, GROUP)
        for tag, cb in (("gden", cb_w), ("gden_uw", cb_uw)):
            W_hat_n = apply_codebook(t, mu, A, Mg, grouped, cb, K, N)
            W_hat = W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * m
            e = eq.eout_sq(W_hat, W_comp, H)
            tot[tag] += e
            if e < e_rtn:
                wins[tag] += 1
            if tag == "gden":
                e_g = e
            else:
                e_guw = e
        # compand (killed A7)
        W_hat_cp_n, gamma = eq.compand_quantize(W_norm, mask, NBITS, GROUP)
        W_cp = W_hat_cp_n * (r.view(-1, 1) * c.view(1, -1)) * m
        e_cp = eq.eout_sq(W_cp, W_comp, H)
        tot["compand"] += e_cp
        if e_cp < e_rtn:
            wins["compand"] += 1
        # saff: survivor-only AFFINE min-max UNIFORM grid at width b (range lever only,
        # no companding, no repack widening). Isolates the pruned-zero range waste.
        q_s, s_s, z_s = eq.repack_quantize(W_norm, m, NBITS, GROUP)
        if s_s.dim() == 3:
            s_s = s_s * r.view(-1, 1, 1)
        else:
            s_s = s_s * r.view(-1, 1)
        s_s = torch.nan_to_num(s_s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
        W_saff = eq.dequant_deployed(q_s, s_s, z_s, m, c)
        e_saff = eq.eout_sq(W_saff, W_comp, H)
        tot["saff"] += e_saff
        if e_saff < e_rtn:
            wins["saff"] += 1
        # binc: bin-CENTERED uniform grid (uniform-source optimal), range-scaling alpha grid
        e_binc0 = None
        for a in ALPHAS:
            W_bc_n = bincenter_quantize(W_norm, m, NBITS, GROUP, alpha=a)
            W_bc = W_bc_n * (r.view(-1, 1) * c.view(1, -1)) * m
            e_bc = eq.eout_sq(W_bc, W_comp, H)
            tot[f"binc{a}"] += e_bc
            if e_bc < e_rtn:
                wins[f"binc{a}"] += 1
            if a == 1.0:
                e_binc0 = e_bc
        tot["rtn"] += e_rtn
        nmat += 1
        # smdz: sign-magnitude + dead-zone (reclaims central wasted codes)
        W_sm_n = eq.sign_magnitude_deadzone_quantize(W_norm, m, NBITS, GROUP)
        W_sm = W_sm_n * (r.view(-1, 1) * c.view(1, -1)) * m
        e_smdz = eq.eout_sq(W_sm, W_comp, H)
        tot["smdz"] += e_smdz
        if e_smdz < e_rtn:
            wins["smdz"] += 1
        print(f"{key:<34}{e_rtn:>12.3e}{e_g/e_rtn:>11.3f}{e_cp/e_rtn:>12.3f}{e_saff/e_rtn:>10.3f}{e_binc0/e_rtn:>12.3f}{e_smdz/e_rtn:>10.3f}", flush=True)

    print(f"\n[POOLED over {nmat} matrices]  sum e_rtn={tot['rtn']:.4e}", flush=True)
    for tag in ["gden", "gden_uw", "compand", "saff", "smdz"] + [f"binc{a}" for a in ALPHAS]:
        print(f"  {tag:<10} sum_e/sum_e_rtn = {tot[tag]/tot['rtn']:.4f}   "
              f"matrices beating RTN: {wins[tag]}/{nmat}", flush=True)
    print("\n[VERDICT] gden mechanism PRESENT iff gden/rtn < 1 pooled AND wins majority "
          "AND it beats compand. If not -> A7 re-confirmed, report null.", flush=True)


if __name__ == "__main__":
    main()
