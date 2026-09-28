#!/usr/bin/env python3
"""MEASURE-FIRST for the hard column-degree-floor candidate (backlog #21). Before building a mask that
guarantees a minimum survivor count per COLUMN, verify the premise: does the deployed soft-beta balanced
mask ACTUALLY starve columns at sp0.6-0.7 in HEALTHY models? If min per-column keep-rate is already
healthy (few/no near-dead columns), a hard floor is a no-op (absorbed) => pivot. If a nontrivial fraction
of columns are near-starved, the floor has room.

Reports, per model over a sample of Linear matrices, at each sparsity:
  - target per-column survivors under EXACT balance = (1-sp)*K
  - distribution of actual per-column survivor counts under the balanced(beta=0.5) mask (min, p1, p5, mean)
  - %columns below 0.5x and 0.25x the exact-balance target (starvation fraction)
  - same for the WANDA mask (baseline reference: balance should starve LESS than wanda if it works)
No eval, no OBS -- just the masks. Fast."""
import os, sys, argparse
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
DEV = "cuda"


def col_stats(mask, sp):
    K, N = mask.shape
    cc = mask.sum(dim=0)                      # [N] survivors per column
    tgt = (1.0 - sp) * K
    return dict(tgt=tgt, cmin=cc.min().item(), cp1=torch.quantile(cc, 0.01).item(),
                cp5=torch.quantile(cc, 0.05).item(), cmean=cc.mean().item(),
                frac_lt_half=(cc < 0.5 * tgt).float().mean().item(),
                frac_lt_qtr=(cc < 0.25 * tgt).float().mean().item(),
                frac_dead=(cc == 0).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsities", default="0.6,0.7")
    ap.add_argument("--n-mats", type=int, default=8, help="sample this many Linear matrices")
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
    sps = [float(s) for s in args.sparsities.split(",")]
    # gather a spread of matrices (early+late layers)
    items = []
    for li in range(len(layers)):
        for ap_ in paths:
            X = acts.get(f'layer_{li}.{ap_}')
            if X is None:
                continue
            mod = layers[li]
            ok = True
            for p in ap_.split('.'):
                if not hasattr(mod, p):
                    ok = False; break
                mod = getattr(mod, p)
            if ok and isinstance(mod, nn.Linear):
                items.append((li, ap_, mod, X))
    step = max(1, len(items) // args.n_mats)
    sample = items[::step][:args.n_mats]
    print(f"# model={args.model} sampled {len(sample)} matrices of {len(items)}", flush=True)
    for sp in sps:
        agg = {k: [] for k in ["cmin", "cp1", "cp5", "frac_lt_half", "frac_lt_qtr", "frac_dead"]}
        aggw = {k: [] for k in agg}
        for (li, ap_, mod, X) in sample:
            W = mod.weight.data.clone().float().to(DEV)
            Xd = X.to(DEV)
            if Xd.dim() == 3:
                Xd = Xd.reshape(-1, Xd.shape[-1])
            Xd = Xd[:min(Xd.shape[0], 256)]
            _, mb = ns.balanced_mask_and_obs(W, Xd, sp, DEV, col_exp=0.5, no_obs=True)
            _, mw = ns.wanda_mask_and_obs(W, Xd, sp, DEV, scope='per_row')
            sb, sw = col_stats(mb, sp), col_stats(mw, sp)
            for k in agg:
                agg[k].append(sb[k]); aggw[k].append(sw[k])
        import statistics as st
        def m(d, k):
            return st.mean(d[k])
        print(f"\n== sp={sp} exact-balance tgt/col varies; fractions are of columns ==", flush=True)
        print(f"  BALANCED(b0.5): cmin/tgt~{m(agg,'cmin'):.1f} frac<0.5tgt={m(agg,'frac_lt_half'):.3f} "
              f"frac<0.25tgt={m(agg,'frac_lt_qtr'):.3f} frac_dead={m(agg,'frac_dead'):.4f}", flush=True)
        print(f"  WANDA        : cmin={m(aggw,'cmin'):.1f} frac<0.5tgt={m(aggw,'frac_lt_half'):.3f} "
              f"frac<0.25tgt={m(aggw,'frac_lt_qtr'):.3f} frac_dead={m(aggw,'frac_dead'):.4f}", flush=True)


if __name__ == "__main__":
    main()
