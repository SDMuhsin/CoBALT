#!/usr/bin/env python3
"""GLOBAL loss-sensitivity profile (attempt-8l). Commission requirement #1 = target GLOBAL model
error, NEVER per-layer. Every prior probe used per-layer tr(D H D^T) (LOCAL). A global-error-targeting
encoding would weight each layer's clip by its GLOBAL loss-sensitivity s_l = E||dLoss/dh_l||^2 (one
backward pass = non-iterative, global; uniform bits => not multiprecision). Precondition: is s_l
non-uniform enough across layers that weighting matters? Report per-layer s_l (normalized) on the DENSE
model, LM loss on a calib batch. If s_l spans >10x, a global-sensitivity-weighted encoding is worth
building; if flat, weighting reduces to uniform (= what awclip already does per-layer).
"""
import argparse, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"


def run_model(MODEL):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    batch = bs.get_calibration_data(tok, n_samples=1, seq_len=256, dataset_key="wikitext2").to(DEV)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map=DEV, low_cpu_mem_usage=True)
    model.eval
    layers = bs.get_transformer_layers(model)
    caps = {}
    hooks = []
    def mk(i):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o.retain_grad; caps[i] = o
        return h
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_hook(mk(i)))
    out = model(batch, labels=batch)
    out.loss.backward
    for h in hooks: h.remove
    sens = []
    for i in range(len(layers)):
        g = caps[i].grad
        sens.append(float((g.float ** 2).mean.item) if g is not None else 0.0)
    s = torch.tensor(sens)
    s = s / s.mean.clamp(min=1e-30)                       # normalize to mean 1
    print(f"\n######## {MODEL} L={len(layers)} loss={float(out.loss):.3f} ########", flush=True)
    print(f"  per-layer GLOBAL sensitivity s_l (normalized mean=1): min={s.min:.3f} max={s.max:.3f} "
          f"span={s.max/s.min.clamp(min=1e-6):.1f}x  std/mean={s.std:.2f}", flush=True)
    prof = "  ".join(f"{v:.2f}" for v in s.tolist)
    print(f"  profile: {prof}", flush=True)
    del model; torch.cuda.empty_cache


def main:
    P = argparse.ArgumentParser; P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    A = P.parse_args
    for m in A.models.split(","): run_model(m.strip)


if __name__ == "__main__":
    main
