#!/usr/bin/env python3
"""Compute per-matrix GLOBAL loss-sensitivity s_m = E|| dLoss/d(output of matrix m) ||^2 (summed
over output channels, averaged over tokens), via ONE backward pass on the DENSE model. Non-iterative,
global. Saved to results/sens_alloc/sm_<model>.pt as {act_key -> float}. act_key = 'layer_{i}.{path}'
matches nosink/collect_activations keys, so the allocator can look up each target matrix's sensitivity.

This is the signal the global-sensitivity sparsity ALLOCATION mask uses (probe_sens_alloc measured it
recovers 98-99% of the optimal water-fill on gemma/tinyllama/qwen; the non-uniform s_m is the lever
probe_global_sens handed off from the encoding effort -> the MASK)."""
import argparse, os, sys
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"


def compute(MODEL, n_samples, seq_len):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                                 device_map=DEV, low_cpu_mem_usage=True)
    model.eval()
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
    caps = {}; hooks = []
    def mk(k):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o.retain_grad(); caps[k] = o
        return h
    for k, mod in targets.items():
        hooks.append(mod.register_forward_hook(mk(k)))
    # accumulate s_m over several calibration sequences (one backward each; sum of grads^2)
    acc = {k: 0.0 for k in targets}
    cnt = 0
    torch.manual_seed(0)
    data = bs.get_calibration_data(tok, n_samples=n_samples, seq_len=seq_len, dataset_key="wikitext2")
    for i in range(data.shape[0]):
        batch = data[i:i + 1].to(DEV)
        model.zero_grad(set_to_none=True)
        caps.clear()
        out = model(batch, labels=batch)
        out.loss.backward()
        for k, o in caps.items():
            g = o.grad
            if g is None:
                continue
            g = g.reshape(-1, g.shape[-1]).float()
            acc[k] += float((g * g).sum(-1).mean())
        cnt += 1
    for hk in hooks:
        hk.remove()
    s_m = {k: acc[k] / max(cnt, 1) for k in targets}
    out_dir = os.path.join(_ROOT, "results", "sens_alloc")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"sm_{MODEL}.pt")
    torch.save(s_m, path)
    vals = torch.tensor([v for v in s_m.values() if v > 0])
    print(f"[{MODEL}] saved {len(s_m)} matrices -> {path}  "
          f"s_m span={float(vals.max()/vals.min()):.1f}x mean={float(vals.mean()):.3e}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    args = ap.parse_args()
    for m in args.models.split(","):
        compute(m.strip(), args.n_samples, args.seq_len)


if __name__ == "__main__":
    main()
