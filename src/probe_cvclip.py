#!/usr/bin/env python3
"""CVCLIP validation (attempt-8, step 1d): does selecting each group's clip rho on an INTERNAL
HELD-OUT FOLD of the calibration set beat awclip's calib-argmin? Grounded in probe_heldout_gap +
probe_offdiag: the per-group rho-argmin overfits the calib fold (~2/3 of awclip's gain is
finite-sample selection overfit). 2-fold CV attacks that overfit directly.

Folds (all wikitext2, disjoint token windows):
  A=[0:16]  build mask/OBS/scale/W_norm + colE_A   (awclip selects on A = same as build data)
  B=[16:32] colE_B                                   (cvclip selects rho on B, independent of A)
Eval Hessians (never used for selection => fair to both arms):
  ptb   cross-distribution held-out
  wikiC=[32:48] in-distribution held-out disjoint from A and B
Both arms use the SAME mask/OBS/scale (built on A); only the rho-SELECTION data differs. If cvclip
retains MORE held-out gain than awclip on >=2 families, CV-selected clipping is real -> implement +
smoke. Non-iterative, global, bpw-identical to RTN (stores same per-group scale/zero).
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

SP, BETA, NBITS, GROUP, CAP = 0.5, 0.5, 3, 128, 256


def colE_of(A, key, device):
    a = A.get(key)
    if a is None:
        return None
    X = a.to(device).float
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], CAP)]
    return (X * X).sum(0).clamp(min=0), X.t @ X


def run_model(MODEL, device="cuda"):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    fold = {"A": collect(model, tok, device, "wikitext2", 16),
            "B": collect(model, tok, device, "wikitext2", 16, take_slice=(16, 32)),
            "C": collect(model, tok, device, "wikitext2", 16, take_slice=(32, 48))}
    try:
        fold["ptb"] = collect(model, tok, device, "ptb", 16)
    except Exception as e:
        print(f"  [warn] ptb {repr(e)[:50]}", flush=True)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache
    EVAL = [k for k in ("ptb", "C") if k in fold]
    lp = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} g={GROUP} eval={EVAL} ########", flush=True)
    mm = [0, 2 ** NBITS - 1]
    ARMS = ["rtn", "awclip", "cvclip"]
    tot = {arm: {ev: 0.0 for ev in EVAL} for arm in ARMS}
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
            if fold["A"].get(key) is None: continue
            W = lin.weight.data.clone.float.to(device); K, N = W.shape
            block = bs._largest_divisor_leq(N, GROUP)
            eA = colE_of(fold["A"], key, device)
            eB = colE_of(fold["B"], key, device)
            if eA is None or eB is None: continue
            colE_A, _ = eA; colE_B, _ = eB
            Hev = {}
            for ev in EVAL:
                r_ = colE_of(fold[ev], key, device)
                Hev[ev] = None if r_ is None else r_[1]
            W_comp, mask = ns.balanced_mask_and_obs(W, fold["A"][key].to(device), SP, device, col_exp=BETA)
            r, c = ns.compute_norm_scales(W_comp, mask, 'col', device); mkf = mask.float
            W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
            rc = r.view(-1, 1) * c.view(1, -1)
            # rtn
            q, s, z, _ = quantize_rtn(W_norm, mm, group_size=block)
            s = (s * r.view(-1, 1, 1)) if s.dim == 3 else (s * r.view(-1, 1))
            s = torch.nan_to_num(s, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
            D_rtn = eq.dequant_deployed(q, s, z, mkf, c) - W
            # awclip: select on colE_A ; cvclip: select on colE_B
            wcolA = (c.float ** 2) * colE_A.float
            wcolB = (c.float ** 2) * colE_B.float
            D_aw = eq.awclip_quantize(W_norm, mkf, NBITS, block, wcolA) * rc * mkf - W
            D_cv = eq.awclip_quantize(W_norm, mkf, NBITS, block, wcolB) * rc * mkf - W
            for ev in EVAL:
                if Hev[ev] is None: continue
                tot["rtn"][ev] += out_err(D_rtn, Hev[ev])
                tot["awclip"][ev] += out_err(D_aw, Hev[ev])
                tot["cvclip"][ev] += out_err(D_cv, Hev[ev])
            nmat += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache
    print(f"  pooled {nmat} matrices. GAIN = 1 - e/e_rtn on each clean held-out Hessian.", flush=True)
    for ev in EVAL:
        gaw = 1 - tot["awclip"][ev] / tot["rtn"][ev]
        gcv = 1 - tot["cvclip"][ev] / tot["rtn"][ev]
        flag = "  <== CVCLIP WINS" if gcv > gaw + 0.003 else ("  (tie)" if abs(gcv - gaw) <= 0.003 else "  (awclip better)")
        print(f"    eval={ev:<4} awclip={gaw*100:+.1f}%  cvclip={gcv*100:+.1f}%  Δ={100*(gcv-gaw):+.2f}pt{flag}", flush=True)


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args
    for m in Aa.models.split(","): run_model(m.strip)


if __name__ == "__main__":
    main
