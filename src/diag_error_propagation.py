#!/usr/bin/env python3
"""CROSS-MODEL error-propagation probe. The static column/activation metrics do NOT isolate gemma-2b's
prune-only collapse (wiki sp0.7 = 5.9e10 vs 125-351 on the other 3). Hypothesis: the collapse is
MODEL-LEVEL error AMPLIFICATION through the residual stream, not a single-matrix distribution property.

Measure, per model: apply the SAME per-row Wanda prune (sp, no quant) used by the collapsing baseline,
then compare dense vs pruned RESIDUAL-STREAM hidden states on a fixed calibration batch. Report per
transformer layer:
  relerr[l] = || h_pruned[l] - h_dense[l] ||_F / || h_dense[l] ||_F   (residual output of block l)
If gemma's relerr grows multiplicatively toward O(1)/explodes while the others stay bounded, THAT is
the mechanism (amplification), and we then localize which layers/sublayers drive it. Verdict is the
end-to-end grids (rule #2); this only explains WHY they collapse."""
import os, sys, argparse, copy
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"


def capture_hidden(model, batch):
    """Return list of per-layer residual-output hidden states [L] each [B,T,H] (cpu float)."""
    caps = {}
    hooks = []
    layers = bs.get_transformer_layers(model)
    def mk(i):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            caps[i] = o.detach().float().cpu()
        return h
    for i, l in enumerate(layers):
        hooks.append(l.register_forward_hook(mk(i)))
    model.eval()
    with torch.no_grad():
        model(batch)
    for h in hooks:
        h.remove()
    return [caps[i] for i in range(len(layers))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, default=0.70)
    ap.add_argument("--n-seq", type=int, default=4)
    ap.add_argument("--ppl", action="store_true", help="also report prune-only wiki PPL (n-ppl seqs)")
    ap.add_argument("--n-ppl", type=int, default=20)
    args = ap.parse_args()
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    batch = cal[:args.n_seq].to(DEV)

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV)
    h_dense = capture_hidden(model, batch)
    del model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV)
    model = bs.apply_wanda_pruning(model, cal, args.sparsity, DEV)
    h_prune = capture_hidden(model, batch)

    print(f"# model={args.model} sp={args.sparsity} n_seq={args.n_seq} L={len(h_dense)}", flush=True)
    print(f"{'layer':>5s} {'relerr':>10s} {'dense_rms':>10s} {'prune_rms':>10s} {'amplif(x_prev)':>14s}", flush=True)
    prev = None
    for l in range(len(h_dense)):
        d, p = h_dense[l], h_prune[l]
        re = (p - d).norm().item() / (d.norm().item() + 1e-20)
        drms = d.pow(2).mean().sqrt().item()
        prms = p.pow(2).mean().sqrt().item()
        amp = (re / prev) if (prev and prev > 1e-9) else float('nan')
        print(f"{l:5d} {re:10.4f} {drms:10.3f} {prms:10.3f} {amp:14.3f}", flush=True)
        prev = re
    ratios = [h_prune[l].pow(2).mean().sqrt().item() / (h_dense[l].pow(2).mean().sqrt().item() + 1e-20)
              for l in range(len(h_dense))]
    relerrs = [(h_prune[l] - h_dense[l]).norm().item() / (h_dense[l].norm().item() + 1e-20)
               for l in range(len(h_dense))]
    # The FINAL layer's ratio is the output-norm and always blips slightly >1 even on graceful models;
    # the causal quantity is the INTERIOR (non-final) sustained ratio (see FINDINGS held-out predictor test).
    interior = max(ratios[:-1]) if len(ratios) > 1 else max(ratios)
    print(f"SUMMARY model={args.model} max_relerr={max(relerrs):.3f} "
          f"max_prune/dense_rms={max(ratios):.3f} interior_max_ratio={interior:.3f} "
          f"amplifies={'YES' if interior > 1.05 else 'no'}", flush=True)
    if args.ppl:
        test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["calibration_seq_len"],
                                n_samples=args.n_ppl, dataset_key="wikitext2")
        ppl = bs.evaluate_perplexity(model, test, DEV)
        print(f"PPL model={args.model} prune_only_wiki_n{args.n_ppl}={ppl:.4g}", flush=True)


if __name__ == "__main__":
    main()
