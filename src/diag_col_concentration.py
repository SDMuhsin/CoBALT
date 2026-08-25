#!/usr/bin/env python3
"""Per-matrix column-concentration screen (adaptive-beta feasibility).

The graceful-model downstream loss of CoBALT-AWQ is on the MASK axis: the column-balanced
mask keeps a survivor set worse for arc than Wanda's, and (mechanism) balance has downstream
upside ONLY where the baseline pruning collapses (interior relerr>1). The proposed lever is a
PER-MATRIX ADAPTIVE beta: balance the matrices Wanda column-STARVES, leave the rest at Wanda.

For adaptive-beta to be mechanically DISTINCT from both global-beta and Wanda, per-matrix
column-concentration must be HETEROGENEOUS. This screen measures, per matrix, how much Wanda's
mask STARVES columns (col_CV of per-column keep-fraction, %dead cols) at the eval sparsity, and
prints the distribution + a gemma-2b reference (where balance is known to help).

One activation pass per model. Proxy only (rule #2: PPL/mask stats RANK; downstream GOVERNS)."""
import os, sys, argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
import nosink as ns  # noqa

DEV = "cuda"


def wanda_mask(W, X, sp, device, scope):
    """Standard Wanda mask (matches wanda-awq baseline). scope: 'global' or 'per_row'."""
    W = W.float().to(device)
    X = X.float().to(device)
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    X = X[:min(X.shape[0], 256)]
    imp = W.abs() * torch.norm(X, dim=0).view(1, -1)
    return ns._threshold_mask(imp, sp, scope=scope)


def colstats(mask):
    kf = mask.mean(0)                       # per-column keep fraction
    cv = (kf.std() / (kf.mean() + 1e-12)).item()
    dead = (kf < 0.05).float().mean().item() * 100
    return cv, dead


def screen(model_key, sp, scope):
    name = bs.MODELS[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, DEV)
    acts = bs.collect_activations(model, cal, DEV)
    layer_paths = bs.get_layer_paths(model)
    rows = []  # (layer, path, type, col_CV, dead%)
    for li, layer in enumerate(ns.get_layers(model)):
        layer = layer.to(DEV)
        for ap in layer_paths:
            parent = layer
            try:
                for p in ap.split('.')[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, ap.split('.')[-1])
            except AttributeError:
                continue
            if not isinstance(linear, nn.Linear):
                continue
            a = acts.get(f'layer_{li}.{ap}', None)
            if a is None:
                continue
            m = wanda_mask(linear.weight.data, a, sp, DEV, scope)
            cv, dead = colstats(m)
            rows.append((li, ap, ns.type_of(ap), cv, dead))
        layer.to("cpu")
    return rows


def summarize(model_key, rows):
    import statistics as st
    cvs = [r[3] for r in rows]
    print(f"\n===== {model_key}: {len(rows)} matrices | col_CV distribution =====")
    print(f"  min {min(cvs):.3f}  p25 {st.quantiles(cvs, n=4)[0]:.3f}  median {st.median(cvs):.3f}  "
          f"p75 {st.quantiles(cvs, n=4)[2]:.3f}  max {max(cvs):.3f}  mean {st.mean(cvs):.3f}")
    # per-type medians (down_proj is the known starvation matrix)
    types = {}
    for r in rows:
        types.setdefault(r[2], []).append(r[3])
    print("  per-type col_CV median | %matrices with col_CV>0.6 (heavily starved):")
    for t in sorted(types):
        v = types[t]
        frac = 100 * sum(x > 0.6 for x in v) / len(v)
        print(f"    {t:10s} n={len(v):3d}  median {st.median(v):.3f}  max {max(v):.3f}  starved%={frac:5.1f}")
    frac_all = 100 * sum(x > 0.6 for x in cvs) / len(cvs)
    print(f"  OVERALL starved% (col_CV>0.6) = {frac_all:.1f}%  "
          f"[heterogeneity: adaptive-beta distinct from Wanda iff this spans a real range]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen-3b,gemma-2b")
    ap.add_argument("--sparsity", type=float, default=0.5)
    ap.add_argument("--scope", default="global", choices=["global", "per_row"])
    args = ap.parse_args()
    for mk in args.models.split(","):
        rows = screen(mk.strip(), args.sparsity, args.scope)
        summarize(mk.strip(), rows)
        import gc; gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
