#!/usr/bin/env python3
"""Cache the per-matrix joint-Fisher INTERACTION M_ij = E[g_i^2 x_j^2] (aligned g,x from the SAME
forward/backward), for the 'balanced_fint' within-matrix mask lever. The NON-separable part of M is the
ONLY signal that can move CoBALT's balanced survivor set (per src/probe_within_mask.py row-absorption
result; FINT shifts 10-16% of survivors vs balanced on gemma/tiny/qwen). Saved per matrix to
results/fisher_int/<model>/<act_key>.pt as fp16 [K,N]. Model-configurable (NOT gemma-only)."""
import argparse, os, sys
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"


def compute(MODEL, n_samples, seq_len, tok_cap):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                                 device_map=DEV, low_cpu_mem_usage=True)
    model.eval
    layers = bs.get_transformer_layers(model)
    paths = bs.get_layer_paths(model)
    targets = {}
    for li, layer in enumerate(layers):
        for p in paths:
            mod = layer; ok = True
            for part in p.split('.'):
                if not hasattr(mod, part):
                    ok = False; break
                mod = getattr(mod, part)
            if ok and isinstance(mod, nn.Linear):
                targets[f'layer_{li}.{p}'] = mod
    caps = {}; cap_in = {}; hooks = []
    def mk(k):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o.retain_grad; caps[k] = o
            cap_in[k] = (inp[0] if isinstance(inp, tuple) else inp).detach
        return h
    for k, mod in targets.items:
        hooks.append(mod.register_forward_hook(mk(k)))
    fint = {k: None for k in targets}
    cnt = 0
    torch.manual_seed(0)
    data = bs.get_calibration_data(tok, n_samples=n_samples, seq_len=seq_len, dataset_key="wikitext2")
    for i in range(data.shape[0]):
        model.zero_grad(set_to_none=True); caps.clear; cap_in.clear
        out = model(data[i:i+1].to(DEV), labels=data[i:i+1].to(DEV)); out.loss.backward
        for k, o in caps.items:
            if o.grad is None:
                continue
            g = o.grad.reshape(-1, o.grad.shape[-1]).float
            x = cap_in[k].reshape(-1, cap_in[k].shape[-1]).float
            m = min(g.shape[0], tok_cap)
            fi = (g[:m] ** 2).t @ (x[:m] ** 2) / m
            fint[k] = fi.cpu if fint[k] is None else fint[k] + fi.cpu
        cnt += 1
    for hk in hooks:
        hk.remove
    out_dir = os.path.join(_ROOT, "results", "fisher_int", MODEL)
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for k, v in fint.items:
        if v is None:
            continue
        torch.save((v / cnt).half, os.path.join(out_dir, k + ".pt"))
        n += 1
    print(f"[{MODEL}] saved {n} interaction matrices -> {out_dir}")


def main:
    ap = argparse.ArgumentParser
    ap.add_argument("--models", default="tinyllama,qwen-1.5b,gemma-2b")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--tok-cap", type=int, default=512)
    args = ap.parse_args
    for m in args.models.split(","):
        compute(m.strip, args.n_samples, args.seq_len, args.tok_cap)


if __name__ == "__main__":
    main
