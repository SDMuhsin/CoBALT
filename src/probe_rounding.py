#!/usr/bin/env python3
"""ROUNDING / GRID-PLACEMENT sweep (attempt-8f, keep-grinding). At a FIXED deployed (scale,zero),
test format-compatible scalar rounding/grid-placement variants NOT yet tried (bpw-identical, global,
non-iterative). Isolates the placement effect. Held-out (ptb) + fit tr(D H D^T) GAIN vs deployed RTN,
balanced vs wanda vs magnitude, 3 families.
  rtn     : deterministic nearest (reference)
  dither  : stochastic rounding (seeded) -- UNBIASED per-weight error; tests whether an unbiased error
            propagates/generalizes better than RTN's systematic bias (the overfit angle).
  offset  : mid-rise grid (levels shifted by half a step).
  gauss   : fixed NON-UNIFORM grid matched to a Gaussian (erfinv level placement) -- reopens the
            value-codebook axis via a fixed shape (survivors uniform => predicted to hurt, but untested).
  twogrid : per-group 1-bit selector between uniform and a skew grid (denser near the larger-|.| side),
            picked by lower WEIGHT-MSE (calibration-free). 1 bit/group ~= 0.008 bpw (counted).
"""
import argparse, os, sys, math
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import eout_quant as eq  # noqa
from sinq.sparse_quant import quantize_rtn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
from probe_heldout_gap import collect, out_err  # noqa
from probe_sscale import magnitude_mask_and_obs  # noqa

SP, BETA, NBITS, GROUP, CAP = 0.5, 0.5, 3, 128, 256
NL = 2 ** NBITS - 1


def _grouped(W_norm, gsize):
    K, N = W_norm.shape
    if N > gsize and N % gsize == 0:
        return W_norm.view(K, N // gsize, gsize), True
    return W_norm.unsqueeze(1), False


def deployed_sz(W_norm, gsize):
    q, s, z, _ = quantize_rtn(W_norm, [0, NL], group_size=gsize)
    return s, z  # s:[K,ng,1], z:[K,ng,1]


def _apply(W_norm, s, z, gsize, mode, gen):
    Wg, gr = _grouped(W_norm, gsize)
    xn = Wg / s + z                                   # grid coords
    if mode == "rtn":
        q = torch.clamp(torch.round(xn), 0, NL); rec = (q - z) * s
    elif mode == "dither":
        fl = torch.floor(xn); frac = xn - fl
        u = torch.rand(xn.shape, generator=gen, device=xn.device)
        q = torch.clamp(fl + (u < frac).float, 0, NL); rec = (q - z) * s
    elif mode == "offset":
        q = torch.clamp(torch.round(xn - 0.5), 0, NL); rec = (q - z + 0.5) * s
    elif mode == "gauss":
        # nonuniform reconstruction levels: erfinv-spaced in [0,NL] mapped to value range
        idx = torch.arange(NL + 1, device=xn.device, dtype=torch.float32)
        p = (idx + 0.5) / (NL + 1)                    # (0,1)
        g = torch.erfinv(2 * p - 1); g = (g - g.min) / (g.max - g.min) * NL  # [0,NL] nonuniform
        # quantize xn to nearest uniform index, then reconstruct at gauss level of that index
        qi = torch.clamp(torch.round(xn), 0, NL).long
        recg = g[qi]                                  # nonuniform code value in [0,NL]
        rec = (recg - z) * s
    elif mode == "twogrid":
        # uniform vs skew (denser near max side): skew index via power curve
        q = torch.clamp(torch.round(xn), 0, NL)
        rec_u = (q - z) * s
        gamma = 0.6
        gi = (NL * (q / NL).clamp(0, 1).pow(gamma))
        rec_s = (gi - z) * s
        # per-group pick lower weight-MSE (calib-free)
        eu = ((Wg - rec_u) ** 2).sum(-1, keepdim=True)
        es = ((Wg - rec_s) ** 2).sum(-1, keepdim=True)
        rec = torch.where(es < eu, rec_s, rec_u)
    else:
        raise ValueError(mode)
    return rec.reshape(W_norm.shape)


def run_model(MODEL, device="cuda"):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    gen = torch.Generator(device=device); gen.manual_seed(1234)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    A = {"fit": collect(model, tok, device, "wikitext2", 16)}
    try:
        A["ptb"] = collect(model, tok, device, "ptb", 16)
    except Exception as e:
        print(f"  [warn] ptb {repr(e)[:40]}", flush=True)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache
    HK = list(A.keys)
    lp = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} g={GROUP} H={HK} ########", flush=True)
    MK = ["balanced", "wanda", "magnitude"]
    ENC = ["rtn", "dither", "offset", "gauss", "twogrid"]
    tot = {mk: {e: {hk: 0.0 for hk in HK} for e in ENC} for mk in MK}
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
                s, z = deployed_sz(W_norm, block)
                for e in ENC:
                    W_hat = _apply(W_norm, s, z, block, e, gen) * rc * mkf
                    D = W_hat - W
                    for hk in HK:
                        if H[hk] is None: continue
                        tot[mk][e][hk] += out_err(D, H[hk])
            nmat += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache
    print(f"  pooled {nmat} matrices/mask. GAIN = 1 - e/e_rtn (>0 = beats deployed RTN).", flush=True)
    for mk in MK:
        row = f"    {mk:<10}"
        for e in ["dither", "offset", "gauss", "twogrid"]:
            for hk in HK:
                g = 1 - tot[mk][e][hk] / tot[mk]["rtn"][hk]
                row += f" {e[:4]}[{hk}]={g*100:+.1f}%"
        print(row, flush=True)


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args
    for m in Aa.models.split(","): run_model(m.strip)


if __name__ == "__main__":
    main
