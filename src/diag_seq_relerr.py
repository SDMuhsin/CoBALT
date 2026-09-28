#!/usr/bin/env python3
"""Per-block PROPAGATED error of a compressed model vs its dense twin, on CALIBRATION text (wikitext2,
the text the method saw) AND on HELD-OUT text (ptb, never seen). Diagnoses the CoBALT-seq arms:
  * seq lowers calib relerr but not held-out relerr  -> the per-block selection overfits calibration
  * seq's per-block relerr is lower but the FINAL relerr is not -> greedy/myopic selection
  * seqfix (sequential inputs, fixed beta) worse than cobalt -> OBS-on-compressed-inputs chases the
    prefix error (target should be the DENSE output) -> fix = dense-target sequential compensation
Reports per block: relerr_l = ||h^_l - h_l||_F/||h_l||_F and the amplification ratio rms(h^_l)/rms(h_l)
(Table 9's interior-max ratio). Usage:
  diag_seq_relerr.py --model gemma-2b --sparsity 0.6 --bits 3 --methods cobalt,cobalt-seqfix,cobalt-seq
"""
import argparse, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import camera_bench as cb  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
DEV = "cuda"


def block_outputs(model, batches):
    blocks = bs.get_transformer_layers(model)
    caps = {}
    hooks = []
    for i, blk in enumerate(blocks):
        def mk(i_):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                caps.setdefault(i_, []).append(o.detach().float().cpu())
            return h
        hooks.append(blk.register_forward_hook(mk(i)))
    model.eval()
    with torch.no_grad():
        for b in batches:
            model(b.to(DEV))
    for h in hooks:
        h.remove()
    return [torch.cat(caps[i], 0) for i in range(len(blocks))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--methods", default="cobalt,cobalt-seqfix,cobalt-seq")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=8, help="held-out / calib sequences used for the probe")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "cobalt_seq", "diag_relerr.tsv"))
    args = ap.parse_args()
    cb.MODEL = args.model
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    L = bs.EVAL_CONFIG["calibration_seq_len"]
    calib = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"], seq_len=L, dataset_key="wikitext2")
    ptb = bs.get_calibration_data(tok, n_samples=args.n, seq_len=L, dataset_key="ptb")
    sets = {"calib": [calib[i:i + 1] for i in range(min(args.n, len(calib)))],
            "heldout": [ptb[i:i + 1] for i in range(len(ptb))]}
    dense = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV)
    dense.config.use_cache = False
    ref = {k: block_outputs(dense, v) for k, v in sets.items()}
    del dense; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for meth in args.methods.split(","):
        m = cb.build_model(meth, args.sparsity, args.bits, tok, cobalt_group_size=128, cobalt_beta=args.beta)
        m.config.use_cache = False
        for k, batches in sets.items():
            outs = block_outputs(m, batches)
            rel = [((o - h).norm() / h.norm().clamp(min=1e-20)).item() for o, h in zip(outs, ref[k])]
            # MASSIVE-ACTIVATION diagnostic: per-token relerr median (each token / its own norm) and the
            # Frobenius relerr with the top-1%-norm tokens (sinks) EXCLUDED. If an arm looks good on raw
            # Frobenius but bad here, it is fitting the sink tokens at the expense of ordinary tokens.
            tokrel, nosink = [], []
            for o, h in zip(outs, ref[k]):
                O = o.reshape(-1, o.shape[-1]); Hh = h.reshape(-1, h.shape[-1])
                hn = Hh.norm(dim=1).clamp(min=1e-20)
                pt = (O - Hh).norm(dim=1) / hn
                tokrel.append(pt.median().item())
                keep = hn <= torch.quantile(hn, 0.99)
                nosink.append(((O[keep] - Hh[keep]).norm() / Hh[keep].norm().clamp(min=1e-20)).item())
            amp = [(o.pow(2).mean().sqrt() / h.pow(2).mean().sqrt().clamp(min=1e-20)).item() for o, h in zip(outs, ref[k])]
            interior = max(amp[:-1]) if len(amp) > 1 else amp[0]
            line = (f"{args.model}\t{meth}\tbits={args.bits}\tsp={args.sparsity}\t{k}\tfinal_relerr={rel[-1]:.4f}\t"
                    f"final_tokmed={tokrel[-1]:.4f}\tfinal_nosink={nosink[-1]:.4f}\t"
                    f"max_relerr={max(rel):.4f}\tinterior_max_amp={interior:.4f}\tper_block=" + ",".join(f"{r:.3f}" for r in rel)
                    + "\ttokmed=" + ",".join(f"{r:.3f}" for r in tokrel) + "\tnosink=" + ",".join(f"{r:.3f}" for r in nosink))
            print(line, flush=True)
            with open(args.out, "a") as f:
                f.write(line + "\n")
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
