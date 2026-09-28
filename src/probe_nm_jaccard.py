#!/usr/bin/env python3
"""MEASURE-FIRST gate 2 for the 2:4 route: does CoBALT's column-balance term still change WHICH 2-of-4
survive once the support is N:M constrained? Under 2:4 the per-row quantile rescale is a no-op (it cannot
reorder entries inside a row's own 4-group), so the column-quantile reweighting (beta) is the ONLY live
CoBALT degree of freedom. Reports, over sampled real matrices of a model:
  * Jaccard(cobalt-2:4, wanda-2:4)  -- ~0.95+ = absorbed (dead), ~0.3 = far-from-magnitude (collapse risk)
  * column keep-rate CV + dead-column fraction for both (is starvation real under 2:4, does balance fix it)
  * Jaccard(cobalt-2:4, cobalt-unstructured@0.5) -- how far the hardware constraint moves the survivor set
"""
import os, sys, argparse
import torch
import torch.nn as nn
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
DEV = "cuda"


def _jacc(a, b):
    a = a.bool(); b = b.bool()
    return ((a & b).sum() / (a | b).sum().clamp(min=1)).item()


def _colstats(m):
    c = m.sum(0)
    return (c.std() / (c.mean() + 1e-9)).item(), (c == 0).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n-mats", type=int, default=12)
    args = ap.parse_args()
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV)
    acts = bs.collect_activations(model, cal, DEV)
    layers = ns.get_layers(model); paths = bs.get_layer_paths(model)
    items = []
    for li in range(len(layers)):
        for ap_ in paths:
            X = acts.get(f'layer_{li}.{ap_}')
            if X is None:
                continue
            mod = layers[li]; ok = True
            for p in ap_.split('.'):
                if not hasattr(mod, p):
                    ok = False; break
                mod = getattr(mod, p)
            if ok and isinstance(mod, nn.Linear):
                items.append((li, ap_, mod, X))
    step = max(1, len(items) // args.n_mats)
    sample = items[::step][:args.n_mats]
    rows = []
    for (li, ap_, mod, X) in sample:
        W = mod.weight.data.clone().float().to(DEV)
        Xd = X.to(DEV)
        if Xd.dim() == 3:
            Xd = Xd.reshape(-1, Xd.shape[-1])
        Xd = Xd[:min(Xd.shape[0], 256)]
        ns.NM_PATTERN = None
        _, mb_u = ns.balanced_mask_and_obs(W, Xd, 0.5, DEV, col_exp=args.beta, no_obs=True)
        ns.NM_PATTERN = (2, 4)
        _, mb = ns.balanced_mask_and_obs(W, Xd, 0.5, DEV, col_exp=args.beta, no_obs=True)
        _, mw = ns.wanda_mask_and_obs(W, Xd, 0.5, DEV)
        ns.NM_PATTERN = None
        cvb, db = _colstats(mb); cvw, dw = _colstats(mw)
        rows.append((li, ap_, _jacc(mb, mw), _jacc(mb, mb_u), cvb, cvw, db, dw))
        print(f"L{li:02d} {ap_:22s} J(cob24,wan24)={rows[-1][2]:.3f} J(cob24,cobU)={rows[-1][3]:.3f} "
              f"colCV cob={cvb:.3f} wan={cvw:.3f} dead cob={db:.4f} wan={dw:.4f}", flush=True)
    import statistics as st
    print(f"# model={args.model} beta={args.beta} n={len(rows)} 2:4")
    print(f"Jaccard(cobalt-2:4, wanda-2:4)       = {st.mean(r[2] for r in rows):.4f} "
          f"(min {min(r[2] for r in rows):.3f} max {max(r[2] for r in rows):.3f})")
    print(f"Jaccard(cobalt-2:4, cobalt-unstruct) = {st.mean(r[3] for r in rows):.4f}")
    print(f"col keep-rate CV: cobalt-2:4={st.mean(r[4] for r in rows):.4f} wanda-2:4={st.mean(r[5] for r in rows):.4f}")
    print(f"dead-col frac:    cobalt-2:4={st.mean(r[6] for r in rows):.5f} wanda-2:4={st.mean(r[7] for r in rows):.5f}")


if __name__ == "__main__":
    main()
