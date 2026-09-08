#!/usr/bin/env python3
"""STRUCTURAL-EXTREME clip (attempt-8n, keep-trying, attacks the closure). Attempt-7 found CoBALT's
mask puts the per-group magnitude-EXTREME in a below-median-||X|| column ~95% of the time -- but never
compared balanced vs wanda. If that structural property is BALANCE-SPECIFIC, a clip based on the BINARY
structural signal (extreme-in-low-||X||-column: mask-driven, robust, NOT a fitted energy => overfit-
immune) could be a CoBALT-specific, non-overfitting scale lever -- escaping the 'activation-aware =>
universal+overfit' branch of the dichotomy.

Measures, per (row,group), balanced vs wanda vs magnitude, 3 families:
 (1) P(extreme in below-group-median ||X|| column) -- is it balance-specific?
 (2) 'sclip' held-out(ptb) output-error GAIN vs RTN: when the row-group max-abs sits in a low-||X||
     column, drop it from the scale (set scale by 2nd-max; the extreme saturates to the grid edge),
     giving the high-||X|| center a finer grid. Binary structural rule, bpw-identical. Balance-specific
     if bal >> wan.
"""
import argparse, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import eout_quant as eq  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
from probe_heldout_gap import collect, out_err  # noqa
from probe_sscale import magnitude_mask_and_obs  # noqa

SP, BETA, NBITS, GROUP, CAP = 0.5, 0.5, 3, 128, 256
NL = 2 ** NBITS - 1


def analyze(W_norm, mask, colnorm, gsize):
    """Return (P_extreme_low, D_rtn_recon, D_sclip_recon) in W_norm space (pre rc/mask)."""
    K, N = W_norm.shape
    if not (N > gsize and N % gsize == 0):
        return None
    ng = N // gsize
    Wg = W_norm.view(K, ng, gsize); Mg = mask.view(K, ng, gsize).bool
    cn = colnorm.view(1, ng, gsize)                             # per-column ||X|| broadcast
    absW = torch.where(Mg, Wg.abs, torch.full_like(Wg, -1.0))
    amax_idx = absW.argmax(-1, keepdim=True)                    # [K,ng,1] extreme position per (row,grp)
    # group median ||X|| over SURVIVOR columns (approx via masked median -> use masked mean as robust proxy)
    cnt = Mg.sum(-1, keepdim=True).clamp(min=1.0)
    cn_surv = torch.where(Mg, cn.expand_as(Wg), torch.full_like(Wg, float('nan')))
    med = torch.nanmedian(cn_surv, dim=-1, keepdim=True).values
    ext_colnorm = torch.gather(cn.expand_as(Wg), -1, amax_idx)  # ||X|| of the extreme's column
    ext_low = (ext_colnorm < med)                              # [K,ng,1] bool: extreme in low-||X|| col
    valid = Mg.any(-1, keepdim=True)
    p_low = float((ext_low & valid).float.sum / valid.float.sum.clamp(min=1))
    # RTN recon (survivor min/max)
    def recon(exclude_extreme):
        big = torch.finfo(torch.float32).max
        m = Mg.clone
        if exclude_extreme is not None:
            drop = exclude_extreme & Mg.any(-1, keepdim=True)
            # remove the extreme position from the scale set where drop
            ar = torch.arange(gsize, device=Wg.device).view(1, 1, -1)
            is_ext = (ar == amax_idx)
            m = Mg & ~(is_ext & drop)
        wmin = torch.where(m, Wg, torch.full_like(Wg, big)).amin(-1, keepdim=True)
        wmax = torch.where(m, Wg, torch.full_like(Wg, -big)).amax(-1, keepdim=True)
        empty = ~m.any(-1, keepdim=True)
        wmin = torch.where(empty, torch.zeros_like(wmin), wmin); wmax = torch.where(empty, torch.zeros_like(wmax), wmax)
        s = ((wmax - wmin) / NL).clamp(min=1e-8); z = -torch.round(wmin / s)
        q = torch.clamp(torch.round(Wg / s + z), 0, NL)
        return ((q - z) * s).reshape(K, N)
    W_rtn = recon(None)
    W_sclip = recon(ext_low)          # drop extreme from scale where it's in a low-||X|| column
    return p_low, W_rtn, W_sclip


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
        print(f"  [warn] ptb {repr(e)[:40]}", flush=True)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache
    HKEY = "ptb" if "ptb" in A else "fit"
    lp = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} g={GROUP} H={HKEY} ########", flush=True)
    MK = ["balanced", "wanda", "magnitude"]
    acc = {mk: {"p": 0.0, "np": 0, "rtn": 0.0, "sclip": 0.0} for mk in MK}
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
            Xh = A[HKEY][key].to(device).float
            if Xh.dim == 3: Xh = Xh.reshape(-1, Xh.shape[-1])
            Xh = Xh[:min(Xh.shape[0], CAP)]
            H = Xh.t @ Xh
            Xf = A["fit"][key].to(device)
            colnorm = torch.norm(Xf.float.reshape(-1, N), dim=0)   # ||X|| per input col (fit)
            for mk in MK:
                if mk == "balanced":
                    W_comp, mask = ns.balanced_mask_and_obs(W, Xf, SP, device, col_exp=BETA)
                elif mk == "wanda":
                    W_comp, mask = ns.wanda_mask_and_obs(W, Xf, SP, device, scope='per_row')
                else:
                    W_comp, mask = magnitude_mask_and_obs(W, Xf, SP, device)
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device); mkf = mask.float
                rc = r.view(-1, 1) * c.view(1, -1)
                W_norm = W_comp / rc
                res = analyze(W_norm, mkf, colnorm, block)
                if res is None: continue
                p_low, W_rtn, W_sclip = res
                acc[mk]["p"] += p_low; acc[mk]["np"] += 1
                acc[mk]["rtn"] += out_err(W_rtn * rc * mkf - W, H)
                acc[mk]["sclip"] += out_err(W_sclip * rc * mkf - W, H)
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache
    print(f"  P(extreme in low-||X|| col) and sclip held-out({HKEY}) GAIN vs RTN:", flush=True)
    for mk in MK:
        a = acc[mk]; n = max(1, a["np"])
        g = 1 - a["sclip"] / a["rtn"] if a["rtn"] > 0 else 0.0
        print(f"    {mk:<10} P_low={a['p']/n*100:.1f}%  sclip_gain={g*100:+.2f}%", flush=True)
    gb = 1 - acc["balanced"]["sclip"]/acc["balanced"]["rtn"]; gw = 1 - acc["wanda"]["sclip"]/acc["wanda"]["rtn"]
    print(f"    Δ(bal-wan) sclip = {(gb-gw)*100:+.2f}pt  {'<== balance-SPECIFIC' if gb>gw+0.02 else '(balance-agnostic)'}", flush=True)


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args
    for m in Aa.models.split(","): run_model(m.strip)


if __name__ == "__main__":
    main
