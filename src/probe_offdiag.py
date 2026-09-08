#!/usr/bin/env python3
"""OFF-DIAGONAL precondition (attempt-8, AXIS E rounding). Coordinated / noise-shaped rounding
can only help if RTN's survivor error D=W_hat-W carries CORRELATED (off-diagonal) output-error
mass that independent nearest-rounding leaves on the table. Per output row k:
    d_k^T H d_k = sum_j d_kj^2 H_jj   (DIAG, irreducible by round-direction alone)
                + sum_{j!=l} d_kj d_kl H_jl   (OFF-DIAG, the only part coordination can move)
offdiag_share = 1 - diag_mass/total. Measured on fit(calib) AND ptb(held-out) Hessians, balanced
vs wanda mask, 3 families. Large & positive-reducible offdiag_share => rounding-coordination has
headroom; near-diagonal => AXIS E is empty (move to grouping AXIS F). Also report the awclip-scaled
D (post-clip) since that is the deployed operating point.
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
from probe_heldout_gap import collect  # noqa

SP, BETA, NBITS, GROUP, CAP = 0.5, 0.5, 3, 128, 256


def masses(D, H):
    """return (total tr(D H D^T), diag mass sum_k sum_j d_kj^2 H_jj)."""
    D = D.double; Hd = H.double
    total = float((D @ Hd * D).sum.item)
    diagH = Hd.diag
    diag = float(((D * D) * diagH.view(1, -1)).sum.item)
    return total, diag


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
    # acc[mask][enc][hk] = [total, diag]
    ENC = ["rtn", "awclip"]; MK = ["balanced", "wanda"]
    acc = {m: {e: {hk: [0.0, 0.0] for hk in HK} for e in ENC} for m in MK}
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
            for m in MK:
                if m == "balanced":
                    W_comp, mask = ns.balanced_mask_and_obs(W, A["fit"][key].to(device), SP, device, col_exp=BETA)
                else:
                    W_comp, mask = ns.wanda_mask_and_obs(W, A["fit"][key].to(device), SP, device, scope='per_row')
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device); mkf = mask.float
                W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
                q, s, z, _ = quantize_rtn(W_norm, mm, group_size=block)
                s = (s * r.view(-1, 1, 1)) if s.dim == 3 else (s * r.view(-1, 1))
                s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
                D_rtn = eq.dequant_deployed(q, s, z, mkf, c) - W
                W_ac = eq.awclip_only(W_comp, mask, NBITS, block, r, c, colE)
                D_ac = W_ac - W
                for hk in HK:
                    if H[hk] is None: continue
                    for e, D in (("rtn", D_rtn), ("awclip", D_ac)):
                        t, d = masses(D, H[hk])
                        acc[m][e][hk][0] += t; acc[m][e][hk][1] += d
            nmat += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache
    print(f"  pooled {nmat} matrices. offdiag_share = 1 - diag/total (fraction of output error", flush=True)
    print(f"  that only coordinated rounding can reduce; ~0 => rounding axis empty).", flush=True)
    for m in MK:
        for e in ENC:
            row = f"    {m:<9} {e:<7}"
            for hk in HK:
                t, d = acc[m][e][hk]
                share = 1 - d / t if t > 0 else 0.0
                row += f"  offdiag[{hk}]={share*100:+.1f}%"
            print(row, flush=True)


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args
    for mm_ in Aa.models.split(","): run_model(mm_.strip)


if __name__ == "__main__":
    main
