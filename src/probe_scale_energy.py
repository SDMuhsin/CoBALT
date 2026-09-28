#!/usr/bin/env python3
"""MEASUREMENT PROBE (attempt-7, step 1): where does CoBALT's OUTPUT error live?

The prior foreclosure (attempt-6) rests on ONE measurement: the per-group-normalized
survivor VALUE law is near-uniform (kurt~1.9) => uniform group-RTN is MSE-optimal for
the WEIGHTS. But downstream error is NOT weight-MSE; it is tr(D H D^T), H=X^T X. Every
prior arm (gden/binc/bincg/compand/smdz) reshaped LEVELS inside RTN's per-group affine
frame while INHERITING RTN's max-abs SCALE, and was judged by the (uniform) marginal.

This probe characterizes, for the DEPLOYED CoBALT pipeline (balanced mask beta, OBS,
col-norm, endpoint group-RTN) on 3 model families, the things that actually drive the
OUTPUT error and were never measured:

 A. MARGINAL vs OUTPUT-ENERGY-WEIGHTED survivor law. Pooled |t| (within-group hull
    position in [-1,1]) unweighted AND weighted by each survivor's output-error
    sensitivity s_ij = (group_step * c_j)^2 * ||X_j||^2. If the ENERGY law concentrates
    at |t|->1 while the plain law is uniform, the extremes (=the SCALE) drive output
    error even though the value shape looks uniform => the scale/clip axis is live.

 B. OUTPUT-ERROR-BY-GRID-POSITION. Bin deployed-RTN survivors by |t|; report each bin's
    share of tr(D H D^T) (diag-H attribution) vs its share of COUNT. Uniform value law +
    uniform error share => RTN optimal. Concentrated error share => a shaped scale/grid
    can move it.

 C. GROUP-SCALE CLIP SWEEP (the untouched axis, DECISION). Re-quantize survivors with the
    group hull scaled by rho in {1.10..0.80} (rho<1 = clip: shrink hull, protect bulk
    granularity, clamp extremes; rho>1 = expand). Measure pooled tr(D H D^T) vs RTN, per
    model. A single GLOBAL rho<1 that reduces OUTPUT error on ALL 3 families = a real,
    novel, bpw-free (scale already stored) lever that no prior arm tested.

 D. ACTIVATION-GATED CLIP. The optimal clip should depend on whether the group's EXTREME
    survivor sits in a low- or high-energy column (clipping a low-energy extreme is free).
    Compare a global deterministic gated clip vs the best uniform rho vs RTN.

 E. OBS-INFLATION MECHANISM. Is the group max-abs of the OBS-COMPENSATED survivors
    inflated vs the raw |W| survivors (residual amplification)? Is that extreme in a
    low-energy column? This is the CoBALT-specific (OBS-driven) hook, absent from AWQ/
    Wanda baselines => bears on S4 attribution.

Runs sp0.5 beta0.5 3-bit, group 64 AND 128, 3 sampled layers x {gemma-2b, tinyllama,
qwen-1.5b}. Calibration-output error only (mechanism existence); held-out is the later cell.
"""
import argparse
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

SP = 0.5
BETA = 0.5
NBITS = 3
CAP256 = 256


def endpoint_rtn_clip(W_norm, mask, nbits, gsize, rho):
    """Deployed endpoint group-RTN but with the per-group hull scaled by rho.
    rho=1 reproduces deployed RTN (up to the min-scale clamp). rho<1 shrinks the
    coded range (clip extremes, finer bulk step); rho>1 expands it. Same 2 stored
    values/group (scale,zero) => bpw-identical. Returns W_hat_norm[K,N]."""
    K, N = W_norm.shape
    m = mask.bool()
    n_levels = 2 ** nbits - 1
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
    half = (0.5 * (w_max - w_min)).clamp(min=1e-8) * rho          # rho scales the coded half-range
    lo = mid - half
    scale = (2.0 * half) / n_levels
    zero = -torch.round(lo / scale)
    q = torch.clamp(torch.round(Wg / scale + zero), 0, n_levels)
    W_hat = (q - zero) * scale
    return W_hat.reshape(K, N)


def endpoint_rtn_gated_clip(W_norm, mask, nbits, gsize, colE, rho_lo, rho_hi):
    """Per-group clip gated on the extreme survivor's column energy: if the group's
    max-|w-mid| survivor sits in a BELOW-median-energy column, clip hard (rho_lo);
    else clip gently (rho_hi). Deterministic global rule keyed on ||X||^2 (already
    collected for OBS). Same storage as RTN. Returns W_hat_norm[K,N]."""
    K, N = W_norm.shape
    m = mask.bool()
    n_levels = 2 ** nbits - 1
    grouped = N > gsize and N % gsize == 0
    if grouped:
        Wg = W_norm.view(K, N // gsize, gsize)
        Mg = m.view(K, N // gsize, gsize)
        cg = colE.view(1, N // gsize, gsize).expand(K, N // gsize, gsize)
    else:
        Wg = W_norm.unsqueeze(1); Mg = m.unsqueeze(1)
        cg = colE.view(1, 1, N).expand(K, 1, N)
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
    w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    mid0 = 0.5 * (w_min + w_max)
    dev = torch.where(Mg, (Wg - mid0).abs(), torch.full_like(Wg, -big))
    ext_idx = dev.argmax(-1, keepdim=True)
    ext_E = torch.gather(cg, -1, ext_idx)
    med_E = torch.where(Mg, cg, torch.full_like(cg, big)).median(-1, keepdim=True).values
    rho = torch.where(ext_E < med_E, torch.full_like(mid0, rho_lo), torch.full_like(mid0, rho_hi))
    half = (0.5 * (w_max - w_min)).clamp(min=1e-8) * rho
    lo = mid0 - half
    scale = (2.0 * half) / n_levels
    zero = -torch.round(lo / scale)
    q = torch.clamp(torch.round(Wg / scale + zero), 0, n_levels)
    W_hat = (q - zero) * scale
    return W_hat.reshape(K, N)


def kurt(x):
    x = x.float(); m = x.mean(); s = x.std().clamp(min=1e-12)
    return float(((x - m) / s).pow(4).mean())


def _q(x, qq):
    """quantile with subsample (torch.quantile caps at ~16M elems)."""
    x = x.float().reshape(-1)
    if x.numel() > 8_000_000:
        x = x[torch.randperm(x.numel())[:8_000_000]]
    return float(x.quantile(qq))


def _cap(x, n=300_000):
    """subsample a per-matrix contribution so pooled arrays stay bounded."""
    x = x.reshape(-1)
    if x.numel() > n:
        x = x[torch.randperm(x.numel(), device=x.device)[:n]]
    return x


def run_model(MODEL, gsizes, device="cuda"):
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
    layers = ns.get_layers(model)
    n_layers = len(layers)
    sample_layers = sorted(set([1, n_layers // 2, n_layers - 2]))
    print(f"\n######## {MODEL}  layers={n_layers} sample={sample_layers} "
          f"sp={SP} beta={BETA} nbits={NBITS} ########", flush=True)

    RHOS = [1.10, 1.05, 1.0, 0.95, 0.90, 0.85, 0.80]
    for gsize in gsizes:
        # accumulators
        pooled_abst, pooled_s = [], []                # |t| and energy weight s_ij (survivors)
        e_rtn_tot = 0.0
        e_rho = {r: 0.0 for r in RHOS}
        wins_rho = {r: 0 for r in RHOS}
        e_gate = 0.0; wins_gate = 0
        e_awclip = 0.0; wins_awclip = 0
        nmat = 0
        # grid-position error share bins (by |t|)
        NB = 5
        binshare_err = torch.zeros(NB)
        binshare_cnt = torch.zeros(NB)
        # OBS inflation
        infl_ratios = []; ext_lowE_frac = []

        for li in sample_layers:
            layer = layers[li].to(device)
            for ap in layer_paths:
                parts = ap.split('.'); parent = layer; ok = True
                for p in parts[:-1]:
                    if not hasattr(parent, p):
                        ok = False; break
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
                Hdiag = H.diag().clamp(min=0)
                colE = (Xa * Xa).sum(0).clamp(min=0)          # ||X_j||^2 [N]
                K, N = W_norm.shape
                m = mask.float()

                # deployed RTN (reference)
                q, scales, zeros, _ = quantize_rtn(W_norm, [0, 2 ** NBITS - 1], group_size=gsize)
                if scales.dim() == 3:
                    scales = scales * r.view(-1, 1, 1)
                else:
                    scales = scales * r.view(-1, 1)
                scales = torch.nan_to_num(scales, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
                W_rtn = eq.dequant_deployed(q, scales, zeros, m, c)
                e_rtn = eq.eout_sq(W_rtn, W_comp, H)
                e_rtn_tot += e_rtn

                # within-group hull position t and energy weight s_ij
                t, mu, A, Mg, grouped = _hull_t(W_norm, mask, gsize)
                if grouped:
                    A_full = A.expand(K, N // gsize, gsize).reshape(K, N)
                else:
                    A_full = A.expand(K, N)
                step = (A_full * c.view(1, -1)) / (2 ** NBITS - 1)     # decode step per position
                s_ij = (step * step) * colE.view(1, -1)                # output-error sensitivity
                mb = mask.bool()
                _idx = None
                _at_m = t.reshape(K, N)[mb].abs()
                _s_m = s_ij[mb]
                if _at_m.numel() > 300_000:
                    _idx = torch.randperm(_at_m.numel(), device=_at_m.device)[:300_000]
                    _at_m = _at_m[_idx]; _s_m = _s_m[_idx]
                pooled_abst.append(_at_m.detach().cpu())
                pooled_s.append(_s_m.detach().cpu())

                # B: error share by |t| bin (diag-H attribution of deployed RTN)
                D = (W_rtn - W_comp)
                err_ij = (D * D) * Hdiag.view(1, -1)                   # diag output-error attribution
                at = t.reshape(K, N)[mb].abs()
                ei = err_ij[mb]
                bidx = (at * NB).clamp(0, NB - 1e-6).long().cpu()
                binshare_err.index_add_(0, bidx, ei.detach().cpu())
                binshare_cnt.index_add_(0, bidx, torch.ones_like(ei.detach().cpu()))

                # C: uniform clip sweep
                for rho in RHOS:
                    W_hat_n = endpoint_rtn_clip(W_norm, mask, NBITS, gsize, rho)
                    W_hat = W_hat_n * (r.view(-1, 1) * c.view(1, -1)) * m
                    e = eq.eout_sq(W_hat, W_comp, H)
                    e_rho[rho] += e
                    if e < e_rtn - 1e-12:
                        wins_rho[rho] += 1
                # D: gated clip (hard on low-energy extreme, gentle else)
                W_gn = endpoint_rtn_gated_clip(W_norm, mask, NBITS, gsize, colE, rho_lo=0.85, rho_hi=1.0)
                W_g = W_gn * (r.view(-1, 1) * c.view(1, -1)) * m
                e_g = eq.eout_sq(W_g, W_comp, H)
                e_gate += e_g
                if e_g < e_rtn - 1e-12:
                    wins_gate += 1

                # D2: AWCLIP — per-group scale chosen by activation-WEIGHTED survivor error.
                # For each rho in a FIXED global grid, per-group weighted error
                # e_g(rho)=sum_{j in g} wcol_j (W_norm_j - W_hat_j(rho))^2, wcol_j=c_j^2*||X_j||^2
                # (=diag of tr(DHD^T)); pick argmin rho PER GROUP. Deterministic global rule,
                # non-iterative (fixed candidate set), stores only the resulting scale (=RTN cost).
                wcol = (c * c) * colE                            # [N] diag output weight in norm space
                if grouped:
                    ng = N // gsize
                    Wg_n = W_norm.view(K, ng, gsize)
                    Mg_n = mask.view(K, ng, gsize).bool()
                    wcg = wcol.view(1, ng, gsize)
                    best_e = None; best_W = None
                    for rho in [1.0, 0.975, 0.95, 0.925, 0.90, 0.875, 0.85]:
                        Whn = endpoint_rtn_clip(W_norm, mask, NBITS, gsize, rho).view(K, ng, gsize)
                        eg = ((Wg_n - Whn) ** 2 * wcg * Mg_n).sum(-1)         # [K,ng] weighted err
                        if best_e is None:
                            best_e = eg; best_W = Whn.clone()
                        else:
                            take = eg < best_e
                            best_e = torch.where(take, eg, best_e)
                            best_W = torch.where(take.unsqueeze(-1), Whn, best_W)
                    W_awc = best_W.reshape(K, N) * (r.view(-1, 1) * c.view(1, -1)) * m
                    e_awc = eq.eout_sq(W_awc, W_comp, H)
                    e_awclip += e_awc
                    if e_awc < e_rtn - 1e-12:
                        wins_awclip += 1

                # E: OBS inflation — max|survivor| of W_comp vs of raw W at same positions
                Wr = W.float().to(device)
                if grouped:
                    Wc_g = (W_comp).view(K, N // gsize, gsize)
                    Wr_g = Wr.view(K, N // gsize, gsize)
                    Mg2 = mask.view(K, N // gsize, gsize).bool()
                    cg = colE.view(1, N // gsize, gsize).expand(K, N // gsize, gsize)
                    big = torch.finfo(torch.float32).max
                    mc = torch.where(Mg2, Wc_g.abs(), torch.zeros_like(Wc_g)).amax(-1)
                    mr = torch.where(Mg2, Wr_g.abs(), torch.zeros_like(Wr_g)).amax(-1)
                    ratio = (mc / mr.clamp(min=1e-8))
                    infl_ratios.append(ratio[Mg2.any(-1)].detach().cpu())
                    # is the compensated extreme in a low-energy column?
                    dev = torch.where(Mg2, (Wc_g).abs(), torch.full_like(Wc_g, -big))
                    ext_idx = dev.argmax(-1, keepdim=True)
                    ext_E = torch.gather(cg, -1, ext_idx).squeeze(-1)
                    med_E = torch.where(Mg2, cg, torch.full_like(cg, big)).median(-1).values
                    good = Mg2.any(-1)
                    ext_lowE_frac.append((ext_E[good] < med_E[good]).float().detach().cpu())
                nmat += 1
            layers[li] = layer.to("cpu")
            torch.cuda.empty_cache()

        # ---- report for this gsize
        AT = torch.cat(pooled_abst); S = torch.cat(pooled_s)
        Sn = S / S.sum().clamp(min=1e-30)
        # energy-weighted mean |t| vs unweighted
        ew_meanabs = float((AT * Sn).sum())
        print(f"\n-- gsize={gsize} pooled survivors n={AT.numel()} nmat={nmat}", flush=True)
        print(f"   |t| law: unweighted mean={float(AT.mean()):.4f} p90={_q(AT,0.90):.4f}  "
              f"ENERGY-weighted mean|t|={ew_meanabs:.4f}  (uniform |t| mean=0.5; "
              f">0.5 => energy at extremes)", flush=True)
        es = binshare_err / binshare_err.sum().clamp(min=1e-30)
        cs = binshare_cnt / binshare_cnt.sum().clamp(min=1e-30)
        print(f"   error-share by |t| bin (0..1): err%={[round(float(x)*100,1) for x in es]}  "
              f"cnt%={[round(float(x)*100,1) for x in cs]}", flush=True)
        print(f"   [C] CLIP SWEEP pooled tr(DHD^T)/RTN, wins/{nmat}:", flush=True)
        for rho in RHOS:
            print(f"       rho={rho:<5} ratio={e_rho[rho]/e_rtn_tot:.4f}  wins={wins_rho[rho]}", flush=True)
        print(f"   [D] gated clip(0.85 lowE / 1.0): ratio={e_gate/e_rtn_tot:.4f} wins={wins_gate}/{nmat}", flush=True)
        print(f"   [D2] AWCLIP (act-weighted per-group scale): ratio={e_awclip/e_rtn_tot:.4f} "
              f"wins={wins_awclip}/{nmat}   <-- the measured lever", flush=True)
        IR = torch.cat(infl_ratios); EL = torch.cat(ext_lowE_frac)
        print(f"   [E] OBS inflation max|Wcomp|/max|Wraw| per group: median={float(IR.median()):.3f} "
              f"p90={_q(IR,0.90):.3f} frac>1.05={float((IR>1.05).float().mean()):.3f} | "
              f"compensated-extreme-in-lowE-col frac={float(EL.mean()):.3f}", flush=True)


def _hull_t(W_norm, mask, gsize):
    K, N = W_norm.shape
    m = mask.bool()
    grouped = N > gsize and N % gsize == 0
    if grouped:
        Wg = W_norm.view(K, N // gsize, gsize); Mg = m.view(K, N // gsize, gsize)
    else:
        Wg = W_norm.unsqueeze(1); Mg = m.unsqueeze(1)
    big = torch.finfo(torch.float32).max
    w_min = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    w_max = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    w_min = torch.where(empty, torch.zeros_like(w_min), w_min)
    w_max = torch.where(empty, torch.zeros_like(w_max), w_max)
    mu = 0.5 * (w_min + w_max)
    A = (0.5 * (w_max - w_min)).clamp(min=1e-8)
    t = ((Wg - mu) / A).clamp(-1, 1)
    return t, mu, A, Mg, grouped


def main():
    P = argparse.ArgumentParser()
    P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    P.add_argument("--gsizes", default="64,128")
    A = P.parse_args()
    gsizes = [int(x) for x in A.gsizes.split(",")]
    for mdl in A.models.split(","):
        run_model(mdl.strip(), gsizes)


if __name__ == "__main__":
    main()
