#!/usr/bin/env python3
"""SURVIVOR-ONLY SCALE (attempt-8b, NEW axis). Deployed CoBALT RTN sets each group's (scale,zero)
from the FULL-group min/max INCLUDING pruned positions (mask applied post-quant, nosink.py:608/616).
BLACKLIST A1's "pruned entries are interior to the group range" holds for MAGNITUDE pruning; CoBALT
prunes by IMPORTANCE |W|*||X|| / row&col quantiles, so a large-|W| low-||X|| weight can be pruned yet
sit at the group magnitude EXTREME -> it inflates the scale and wastes RTN granularity on a value
that gets zeroed anyway.

sscale = compute (scale,zero) over SURVIVORS ONLY. Same dense codes, same bit-width, same decoder =>
bpw-IDENTICAL, NOT repack (no survivor-only codes, no widening). CALIBRATION-FREE (weight+mask only)
=> structurally OVERFIT-IMMUNE, unlike awclip. Potentially CoBALT-mask-SPECIFIC (magnitude/unbalanced
masks prune interior/differently).

Measures, per mask in {balanced(CoBALT), wanda, magnitude}: (1) frac_exterior = fraction of groups
whose survivor hull is strictly INSIDE the full hull (a pruned position sets min or max);
(2) mean survivor_range/full_range; (3) held-out GAIN of sscale (and awclip ref) vs deployed full-
scale RTN on H_fit and H_ptb. Balance-specific if sscale GAIN(balanced) >> GAIN(magnitude).
Overfit-free if GAIN[fit]~=GAIN[ptb]. g128/3bit/sp0.5/beta0.5, 3 families.
"""
import argparse, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import eout_quant as eq  # noqa
from sinq.sparse_quant import quantize_rtn, compute_hessian_inverse  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
from probe_heldout_gap import collect, out_err  # noqa

SP, BETA, NBITS, GROUP, CAP = 0.5, 0.5, 3, 128, 256


def magnitude_mask_and_obs(W, X, sparsity, device):
    """Pure |W| magnitude mask (global top-k), SAME OBS as balanced_mask_and_obs -- the control
    where pruned positions ARE interior (sscale should NOT help)."""
    K = W.shape[0]
    X = X.float
    imp = W.abs
    mask = ns._threshold_mask(imp, sparsity, scope='global')
    H_inv = compute_hessian_inverse(X, damping=None)
    H_inv_diag = H_inv.diag
    W_comp = W.clone
    for i in range(K):
        pruned = W[i] * (1.0 - mask[i])
        comp = -H_inv @ (pruned / H_inv_diag)
        W_comp[i] = W[i] * mask[i] + comp * mask[i]
    return W_comp, mask


def sscale_quantize(W_norm, mask, nbits, gsize):
    K, N = W_norm.shape; m = mask.bool; nl = 2 ** nbits - 1
    grouped = N > gsize and N % gsize == 0
    if grouped:
        ng = N // gsize; Wg = W_norm.view(K, ng, gsize); Mg = m.view(K, ng, gsize)
    else:
        Wg = W_norm.unsqueeze(1); Mg = m.unsqueeze(1)
    big = torch.finfo(torch.float32).max
    wmin = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
    wmax = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
    empty = ~Mg.any(-1, keepdim=True)
    wmin = torch.where(empty, torch.zeros_like(wmin), wmin); wmax = torch.where(empty, torch.zeros_like(wmax), wmax)
    scale = ((wmax - wmin) / nl).clamp(min=1e-8); zero = -torch.round(wmin / scale)
    q = torch.clamp(torch.round(Wg / scale + zero), 0, nl)
    return ((q - zero) * scale).reshape(K, N)


def hull_stats(W_norm, mask, gsize):
    """frac of groups with survivor hull strictly inside full hull, and mean surv_range/full_range."""
    K, N = W_norm.shape; m = mask.bool
    if not (N > gsize and N % gsize == 0):
        return 0.0, 1.0
    ng = N // gsize; Wg = W_norm.view(K, ng, gsize); Mg = m.view(K, ng, gsize)
    big = torch.finfo(torch.float32).max
    fr = Wg.amax(-1) - Wg.amin(-1)
    smin = torch.where(Mg, Wg, torch.full_like(Wg, big)).amin(-1)
    smax = torch.where(Mg, Wg, torch.full_like(Wg, -big)).amax(-1)
    has = Mg.any(-1)
    sr = torch.where(has, smax - smin, fr)
    fr = fr.clamp(min=1e-9)
    ratio = (sr / fr).clamp(0, 1)
    strict = (ratio < 0.999) & has
    return float(strict.float.mean.item), float(ratio[has].mean.item)


def run_model(MODEL, device="cuda"):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    A = {"fit": collect(model, tok, device, "wikitext2", 16)}
    try:
        A["ptb"] = collect(model, tok, device, "ptb", 16)
    except Exception as e:
        print(f"  [warn] ptb {repr(e)[:50]}", flush=True)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache
    HK = list(A.keys)
    lp = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} g={GROUP} H={HK} ########", flush=True)
    mm = [0, 2 ** NBITS - 1]
    MK = ["balanced", "wanda", "magnitude"]
    ENC = ["rtn", "sscale", "awclip"]
    tot = {mk: {e: {hk: 0.0 for hk in HK} for e in ENC} for mk in MK}
    hull = {mk: [0.0, 0.0] for mk in MK}
    nmat = 0
    for li in sample:
        layer = layers[li].to(device)
        for ap in lp:
            parts = ap.split('.'); parent = layer; ok = True
            for p in parts[:-1]:
                if not hasattr(parent, p): ok = False; break
                parent = getattr(parent, p)
            if not ok or not hasattr(parent, parts[-1]): continue
            lin = getattr(parent, parts[-1])
            if not isinstance(lin, torch.nn.Linear): continue
            key = f'layer_{li}.{ap}'
            if A["fit"].get(key) is None: continue
            W = lin.weight.data.clone.float.to(device); K, N = W.shape
            block = bs._largest_divisor_leq(N, GROUP)
            H = {}
            for hk in HK:
                a = A[hk].get(key)
                if a is None: H[hk] = None; continue
                Xh = a.to(device).float
                if Xh.dim == 3: Xh = Xh.reshape(-1, Xh.shape[-1])
                Xh = Xh[:min(Xh.shape[0], CAP)]
                H[hk] = Xh.t @ Xh
            Xf = A["fit"][key].to(device).float
            if Xf.dim == 3: Xf = Xf.reshape(-1, Xf.shape[-1])
            Xf = Xf[:min(Xf.shape[0], CAP)]
            colE = (Xf * Xf).sum(0).clamp(min=0)
            for mk in MK:
                if mk == "balanced":
                    W_comp, mask = ns.balanced_mask_and_obs(W, A["fit"][key].to(device), SP, device, col_exp=BETA)
                elif mk == "wanda":
                    W_comp, mask = ns.wanda_mask_and_obs(W, A["fit"][key].to(device), SP, device, scope='per_row')
                else:
                    W_comp, mask = magnitude_mask_and_obs(W, A["fit"][key].to(device), SP, device)
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device); mkf = mask.float
                rc = r.view(-1, 1) * c.view(1, -1)
                W_norm = W_comp / rc
                fe, rr = hull_stats(W_norm, mask, block)
                hull[mk][0] += fe; hull[mk][1] += rr
                # rtn (full-group scale, deployed)
                q, s, z, _ = quantize_rtn(W_norm, mm, group_size=block)
                s = (s * r.view(-1, 1, 1)) if s.dim == 3 else (s * r.view(-1, 1))
                s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
                D_rtn = eq.dequant_deployed(q, s, z, mkf, c) - W
                # sscale
                D_ss = sscale_quantize(W_norm, mkf, NBITS, block) * rc * mkf - W
                # awclip ref
                D_aw = eq.awclip_only(W_comp, mask, NBITS, block, r, c, colE) - W
                for hk in HK:
                    if H[hk] is None: continue
                    tot[mk]["rtn"][hk] += out_err(D_rtn, H[hk])
                    tot[mk]["sscale"][hk] += out_err(D_ss, H[hk])
                    tot[mk]["awclip"][hk] += out_err(D_aw, H[hk])
            nmat += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache
    print(f"  pooled {nmat} matrices/mask. frac_ext=groups w/ pruned pos at hull; rr=surv/full range.", flush=True)
    print(f"  GAIN = 1 - e/e_rtn_full (deployed). sscale is calib-FREE (overfit-immune).", flush=True)
    for mk in MK:
        fe = hull[mk][0] / nmat; rr = hull[mk][1] / nmat
        row = f"    {mk:<10} frac_ext={fe*100:5.1f}%  rr={rr:.3f}  "
        for e in ["sscale", "awclip"]:
            for hk in HK:
                g = 1 - tot[mk][e][hk] / tot[mk]["rtn"][hk]
                row += f" {e}[{hk}]={g*100:+.1f}%"
        print(row, flush=True)


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args
    for m in Aa.models.split(","): run_model(m.strip)


if __name__ == "__main__":
    main
