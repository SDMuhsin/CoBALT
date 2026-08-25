#!/usr/bin/env python3
"""Leg-(c) test: on gemma-2b, does CoBALT's COLUMN-BALANCE mask (and/or OBS compensation) SUPPRESS the
residual-stream error amplification that makes plain per-row Wanda pruning collapse (relerr peaks 3.4,
prune_rms blows up 3-4x)? Compares, all at sp0.7, prune-only (no quant), the SAME model:
  wanda_perrow (no OBS)      -- the collapsing baseline
  balanced_b0.5 (no OBS)     -- CoBALT's mask alone (col self-normalized quantile), the claimed lever
  wanda_perrow + OBS         -- isolate OBS compensation's effect
  balanced_b0.5 + OBS        -- CoBALT's mask + OBS (its actual prune stage, minus quant)
Reports per-layer relerr vs dense. Whichever variant keeps relerr bounded (<~0.5, no blow-up) is what
CoBALT actually operates on. Verdict remains the end-to-end grids (rule #2)."""
import os, sys, argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
import nosink as ns  # noqa

DEV = "cuda"


def capture_hidden(model, batch):
    caps = {}; hooks = []
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


def relerr_curve(h_dense, h_var):
    return [((h_var[l] - h_dense[l]).norm().item() / (h_dense[l].norm().item() + 1e-20))
            for l in range(len(h_dense))]


def _thr_mask(imp, sparsity, scope):
    K, N = imp.shape
    if scope == 'per_row':
        kp = int(N * sparsity)
        thr = torch.kthvalue(imp, kp, dim=1, keepdim=True).values
        return (imp > thr).float()
    n_prune = int(K * N * sparsity)
    thr = torch.kthvalue(imp.view(-1), n_prune).values
    return (imp.view(-1) > thr).view(K, N).float()


def _cheap_mask(W, anorm, sp, kind):
    imp = W.abs() * anorm.view(1, -1)
    if kind == "wanda":
        return _thr_mask(imp, sp, 'per_row')
    K, N = W.shape
    kr, kc = int(N * sp), int(K * sp)
    if kr > 0:
        qr = torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qr
    if kc > 0:
        qc = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qc.pow(0.5)
    return _thr_mask(imp, sp, 'global')


def prune_model(model, acts, sp, variant):
    """In-place: set each linear's weight to masked (optionally OBS-compensated) weight."""
    obs = variant.endswith("+obs")
    kind = "wanda" if variant.startswith("wanda") else "balanced"
    layers = ns.get_layers(model)
    paths = bs.get_layer_paths(model)
    for li in range(len(layers)):
        layer = layers[li]
        for ap in paths:
            mod = layer; ok = True
            for p in ap.split('.'):
                if not hasattr(mod, p): ok = False; break
                mod = getattr(mod, p)
            if not ok or not isinstance(mod, nn.Linear):
                continue
            X = acts.get(f'layer_{li}.{ap}')
            if X is None:
                continue
            W = mod.weight.data.clone().float().to(DEV)
            Xd = X.to(DEV)
            if Xd.dim() == 3:
                Xd = Xd.reshape(-1, Xd.shape[-1])
            Xd = Xd[:min(Xd.shape[0], 256)]
            if obs:
                # exact CoBALT prune stage (mask + OBS compensation), no quant
                if kind == "wanda":
                    W_comp, mask = ns.wanda_mask_and_obs(W, Xd, sp, DEV, scope='per_row')
                else:
                    W_comp, mask = ns.balanced_mask_and_obs(W, Xd, sp, DEV, col_exp=0.5)
                new_W = W_comp * mask
            else:
                anorm = torch.norm(Xd.float(), dim=0)
                mask = _cheap_mask(W, anorm, sp, kind)
                new_W = W * mask
            mod.weight.data = new_W.to(mod.weight.dtype)
            del Xd
        torch.cuda.empty_cache()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-2b")
    ap.add_argument("--sparsity", type=float, default=0.70)
    ap.add_argument("--variant", required=True,
                    choices=["wanda", "balanced", "wanda+obs", "balanced+obs"])
    ap.add_argument("--n-seq", type=int, default=4)
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
    # collect acts on the SAME (dense) model for the mask/OBS
    acts = bs.collect_activations(model, cal, DEV)
    model = prune_model(model, acts, args.sparsity, args.variant)
    h_var = capture_hidden(model, batch)
    curve = relerr_curve(h_dense, h_var)
    print(f"# model={args.model} sp={args.sparsity} variant={args.variant} L={len(curve)}", flush=True)
    print(f"{'layer':>5s} {'relerr':>10s}", flush=True)
    for l, r in enumerate(curve):
        print(f"{l:5d} {r:10.4f}", flush=True)
    print(f"SUMMARY variant={args.variant} max_relerr={max(curve):.4f} final_relerr={curve[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
