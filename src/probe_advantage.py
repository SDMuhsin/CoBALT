#!/usr/bin/env python3
"""ADVANTAGE-ENCODING probe (attempt-8c). All 6 prior encoding probes measure balanced~=wanda
because both are importance+OBS masks differing ONLY in per-column survivor-count uniformity, and
deficiency-correcting encodings help whoever HAS the deficiency (the unbalanced baseline). The
untested INVERSION: an encoding that exploits an ADVANTAGE balance CREATES.

(1) Measure the structural difference: per-column survivor-count n_j coefficient-of-variation
    CoV(n_j)=std/mean, balanced vs wanda vs magnitude. (balance should give low CoV = uniform.)
(2) Test the first advantage-encoding instance: dcrm = per-column DC removal. Subtract each column's
    SURVIVOR-MEAN m_j (a WEIGHT statistic -> calibration-free, overfit-immune) before group-RTN;
    store m_j per column (bpw +16/K, ~free, same slot class as c); decode adds it back. If columns
    carry per-column DC that varies, centering tightens the per-group range => finer grid. Reliable
    only where per-column stats are stable (uniform n_j = balance). Predict dcrm helps balanced
    held-out, neutral/worse for wanda sparse columns. Report held-out(ptb)+fit GAIN vs deployed RTN.
g128/3bit/sp0.5/beta0.5, 3 families.
"""
import argparse, os, sys
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


def col_count_cov(mask):
    n = mask.float.sum(0)                      # [N] survivors per column
    nz = n[n > 0]
    if nz.numel == 0:
        return 0.0
    return float((nz.std / nz.mean.clamp(min=1e-8)).item)


def dcrm_quantize(W_norm, mask, nbits, gsize):
    """Per-column DC removal + group RTN of the residual. m_j = survivor-mean per column."""
    K, N = W_norm.shape; m = mask.bool
    cnt = m.float.sum(0).clamp(min=1.0)                       # [N]
    mj = torch.where(m, W_norm, torch.zeros_like(W_norm)).sum(0) / cnt   # [N] per-col survivor mean
    Wc = W_norm - mj.view(1, -1)
    q, s, z, _ = quantize_rtn(Wc, [0, 2 ** nbits - 1], group_size=gsize)
    Wc_hat = eq.dequant_deployed(q, s, z, torch.ones_like(W_norm), torch.ones(N, device=W_norm.device))
    return Wc_hat + mj.view(1, -1)                              # add DC back


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
    tot = {mk: {e: {hk: 0.0 for hk in HK} for e in ["rtn", "dcrm"]} for mk in MK}
    cov = {mk: 0.0 for mk in MK}
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
                cov[mk] += col_count_cov(mask)
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device); mkf = mask.float
                rc = r.view(-1, 1) * c.view(1, -1)
                W_norm = W_comp / rc
                q, s, z, _ = quantize_rtn(W_norm, mm, group_size=block)
                s = (s * r.view(-1, 1, 1)) if s.dim == 3 else (s * r.view(-1, 1))
                s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
                D_rtn = eq.dequant_deployed(q, s, z, mkf, c) - W
                D_dc = dcrm_quantize(W_norm, mkf, NBITS, block) * rc * mkf - W
                for hk in HK:
                    if H[hk] is None: continue
                    tot[mk]["rtn"][hk] += out_err(D_rtn, H[hk])
                    tot[mk]["dcrm"][hk] += out_err(D_dc, H[hk])
            nmat += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache
    print(f"  pooled {nmat} matrices/mask. CoV(n_j)=per-column count coeff-of-variation (low=uniform=balance).", flush=True)
    print(f"  dcrm GAIN = 1 - e/e_rtn (per-column DC removal; calib-free). balance-favoring if bal>>wan,mag.", flush=True)
    for mk in MK:
        row = f"    {mk:<10} CoV={cov[mk]/nmat:.3f}  "
        for hk in HK:
            g = 1 - tot[mk]["dcrm"][hk] / tot[mk]["rtn"][hk]
            row += f" dcrm[{hk}]={g*100:+.2f}%"
        print(row, flush=True)


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args
    for m in Aa.models.split(","): run_model(m.strip)


if __name__ == "__main__":
    main
