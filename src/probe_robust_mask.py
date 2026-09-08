#!/usr/bin/env python3
"""MEASURE (lever #16 info-filter): does the CoBALT balanced survivor set CHANGE across calibration
DISTRIBUTIONS (wiki vs ptb vs c4)? If the mask is distribution-sensitive, a ROBUST mask (survivors
important across ALL distributions) could improve HELD-OUT downstream (where CoBALT's edge provably
lives, [[calib-error-blind-to-cobalt]]). If ||X|| is distribution-stable (Jaccard ~1.0), skip the build.
Also reports the potential lever size: Jaccard(mask_wiki, mask_min) where mask_min uses per-column
min over the 3 distributions."""
import argparse, os, sys
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"; SP = 0.6; BETA = 0.5


def kmask(imp, sp):
    K, N = imp.shape
    kr, kc = int(N * sp), int(K * sp)
    imp = imp.clone
    if kr > 0:
        imp = imp / torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
    if kc > 0:
        imp = imp / torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30).pow(BETA)
    thr = torch.kthvalue(imp.reshape(-1), int(K * N * sp)).values
    return (imp.reshape(-1) > thr).view(K, N)


def jac(a, b):
    a = a.bool; b = b.bool
    return float((a & b).sum / (a | b).sum.clamp(min=1))


def run(MODEL, n_calib):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=DEV, low_cpu_mem_usage=True)
    model.eval
    paths = bs.get_layer_paths(model)
    acts = {}
    for dk in ["wikitext2", "ptb", "c4"]:
        try:
            torch.manual_seed(1)
            cal = bs.get_calibration_data(tok, n_samples=n_calib, seq_len=256, dataset_key=dk)
            acts[dk] = bs.collect_activations(model, cal, DEV)
        except Exception as e:
            print(f"[{MODEL}] {dk} calib failed: {e}")
    for _l in ns.get_layers(model):
        _l.to("cpu")
    torch.cuda.empty_cache
    dks = list(acts)
    jwp, jwc, jwmin = [], [], []
    corr_wp = []
    for li, layer in enumerate(bs.get_transformer_layers(model)):
        for p in paths:
            mod = layer; ok = True
            for part in p.split('.'):
                if not hasattr(mod, part):
                    ok = False; break
                mod = getattr(mod, part)
            if not (ok and isinstance(mod, nn.Linear)):
                continue
            key = f'layer_{li}.{p}'
            if any(key not in acts[d] for d in dks):
                continue
            W = mod.weight.data.float.to(DEV)
            xn = {}
            for d in dks:
                X = acts[d][key].float.to(DEV)
                if X.dim == 3:
                    X = X.reshape(-1, X.shape[-1])
                xn[d] = torch.norm(X[:min(X.shape[0], 256)], dim=0)
            m_wiki = kmask(W.abs * xn['wikitext2'].view(1, -1), SP)
            if 'ptb' in xn:
                m_ptb = kmask(W.abs * xn['ptb'].view(1, -1), SP)
                jwp.append(jac(m_wiki, m_ptb))
                a = xn['wikitext2'].argsort.argsort.float; b = xn['ptb'].argsort.argsort.float
                corr_wp.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
            if 'c4' in xn:
                jwc.append(jac(m_wiki, kmask(W.abs * xn['c4'].view(1, -1), SP)))
            xmin = torch.stack([xn[d] for d in dks]).min(0).values
            jwmin.append(jac(m_wiki, kmask(W.abs * xmin.view(1, -1), SP)))
            del W
        layer.to("cpu"); torch.cuda.empty_cache

    def mean(x):
        return sum(x) / len(x) if x else float('nan')
    print(f"\n===== {MODEL} (dists={dks}, sp={SP}) =====")
    print(f"  Jaccard(mask_wiki, mask_ptb) = {mean(jwp):.3f}   col-||X|| rank-corr wiki~ptb = {mean(corr_wp):.3f}")
    print(f"  Jaccard(mask_wiki, mask_c4)  = {mean(jwc):.3f}")
    print(f"  Jaccard(mask_wiki, mask_MIN) = {mean(jwmin):.3f}  ({(1-mean(jwmin))*100:.1f}% survivors differ)")
    ov = mean(jwmin)
    print(f"  >>> robust-min mask {'REDUNDANT (skip)' if ov > 0.95 else 'DIFFERENT (worth a downstream smoke)'}")


def main:
    ap = argparse.ArgumentParser
    ap.add_argument("--models", default="tinyllama,qwen-1.5b")
    ap.add_argument("--n-calib", type=int, default=16)
    args = ap.parse_args
    for m in args.models.split(","):
        try:
            run(m.strip, args.n_calib)
        except Exception as e:
            import traceback; traceback.print_exc; print(f"[{m}] FAIL {e}")


if __name__ == "__main__":
    main
