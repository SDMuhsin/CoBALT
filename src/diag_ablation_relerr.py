#!/usr/bin/env python3
"""MEDIATOR probe for the mask×OBS factorial ablation (gemma-2b).

Measures the residual-stream interior relative error for ONE factorial cell, built with the
EXACT camera_bench.build_model object the accuracy grid uses (so relerr is the same object's
property, not a re-implementation). For a given (method, beta, bits, sparsity):
  relerr[l]  = ||h_comp[l] - h_dense[l]||_F / ||h_dense[l]||_F   (block-l residual output)
  interior_max_ratio = max_{l<L} rms(h_comp[l]) / rms(h_dense[l])   (the established collapse
      predictor — crosses ~1.0 exactly when prune-collapse happens; see diag_error_propagation).
Emits one CSV row: method,bits,sparsity,beta,max_relerr,interior_max_ratio,amplifies.

Usage: python src/diag_ablation_relerr.py --method cobalt-noobs --beta 0.7 --bits 3 \
         --sparsity 0.6 --group-size 128 --csv results/benchmark_ablation_maskobs/relerr.csv
"""
import os, sys, argparse, csv, fcntl
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import camera_bench as cb     # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa


def capture_hidden(model, batch):
    caps, hooks = {}, []
    layers = bs.get_transformer_layers(model)
    def mk(i):
        def h(mod, inp, out):
            caps[i] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
        return h
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_hook(mk(i)))
    model.eval()
    with torch.no_grad():
        model(batch)
    for h in hooks:
        h.remove()
    return [caps[i] for i in range(len(layers))]


def append_row(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.writer(f)
        if new:
            w.writerow(["method", "bits", "sparsity", "beta", "group_size",
                        "max_relerr", "interior_max_ratio", "final_ratio", "amplifies", "L"])
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2b")
    ap.add_argument("--method", required=True)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--sparsity", type=float, default=0.6)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-seq", type=int, default=4)
    ap.add_argument("--csv", default=os.path.join(_ROOT, "results", "benchmark_ablation_maskobs", "relerr.csv"))
    args = ap.parse_args()

    cb.MODEL = args.model
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    batch = cal[:args.n_seq].to(cb.DEV)

    dense = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(cb.DEV)
    h_dense = capture_hidden(dense, batch)
    del dense; torch.cuda.empty_cache()

    model = cb.build_model(args.method, args.sparsity, args.bits, tok,
                           cobalt_group_size=args.group_size, cobalt_beta=args.beta)
    h_comp = capture_hidden(model, batch)
    del model; torch.cuda.empty_cache()

    L = len(h_dense)
    relerrs = [(h_comp[l] - h_dense[l]).norm().item() / (h_dense[l].norm().item() + 1e-20) for l in range(L)]
    ratios = [h_comp[l].pow(2).mean().sqrt().item() / (h_dense[l].pow(2).mean().sqrt().item() + 1e-20) for l in range(L)]
    interior = max(ratios[:-1]) if L > 1 else max(ratios)
    print(f"# method={args.method} beta={args.beta} bits={args.bits} sp={args.sparsity} L={L}", flush=True)
    for l in range(L):
        print(f"  layer {l:2d} relerr={relerrs[l]:.4f} rms_ratio={ratios[l]:.4f}", flush=True)
    print(f"SUMMARY method={args.method} beta={args.beta} bits={args.bits} sp={args.sparsity} "
          f"max_relerr={max(relerrs):.4f} interior_max_ratio={interior:.4f} "
          f"amplifies={'YES' if interior > 1.05 else 'no'}", flush=True)
    append_row(args.csv, [args.method, args.bits, f"{args.sparsity:.2f}", f"{args.beta:g}", args.group_size,
                          f"{max(relerrs):.6f}", f"{interior:.6f}", f"{ratios[-1]:.6f}",
                          "YES" if interior > 1.05 else "no", L])


if __name__ == "__main__":
    main()
