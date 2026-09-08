#!/usr/bin/env python3
"""HELD-OUT reconstruction SCREEN for mask candidates (the correct screen -- calib error is blind to
CoBALT's edge, [[calib-error-blind-to-cobalt]], [[quantizer-attempt8-heldout-gap]]). Fit each mask+OBS on
WIKI activations, then measure reconstruction error on DISJOINT PTB activations:
    err_ho = || X_c4 (Wcomp - W)^T ||_F^2 / || X_c4 W^T ||_F^2   (relative held-out output error)
VALIDATION: base cobalt (balanced) should have LOWER held-out err than wanda (that IS the measured edge).
If so, the screen tracks the downstream edge -> use it to rank NEW candidate masks cheaply (only
screen-WINNERS get an expensive downstream smoke). Prune+OBS only (no quant; quant is a universal step).
Non-iterative per matrix. sp default 0.6 (the confirmed healthy collapse-onset point)."""
import os, sys, argparse
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
DEV = "cuda"


def obs_from_mask(W, X, mask):
    """OBS-compensate given a fixed keep-mask, on activations X. Returns Wcomp (survivors compensated)."""
    H_inv = ns.compute_hessian_inverse(X, damping=None); H_inv_diag = H_inv.diag
    P = (W * (1.0 - mask)) / H_inv_diag.view(1, -1)
    return (W - P @ H_inv) * mask


def _bal_from_imp(imp, sp, beta):
    """balanced global-top-k keep-mask from an arbitrary importance map (row+col quantile norm)."""
    return ns.balanced_keepmask_local(imp, sp, beta)


def build_masks(W, Xw, sp, beta=0.5):
    """Return {name: keep-mask} for all screened candidates, all fit on WIKI Xw. Screening NEW principles
    that might LOWER held-out error below base balanced (=beat the downstream edge)."""
    K, N = W.shape
    l2 = torch.norm(Xw, dim=0).view(1, -1)                    # L2 col norm (base, outlier-dominated)
    imp = W.abs * l2
    out = {}
    out['balanced'] = _bal_from_imp(imp, sp, beta)            # base cobalt
    _, out['wanda'] = ns.wanda_mask_and_obs(W, Xw, sp, DEV, scope='per_row')
    out['exact'] = ns.exact_doubly_balanced_mask(imp, sp)     # known-bad sanity
    # --- NEW candidates targeting held-out generalization ---
    # (a) ROBUST column stat: mean|X| instead of L2 (L2 is dominated by outlier tokens that differ across
    #     distributions => L2 mask overfits calib; mean|X| is a more transferable channel-energy stat)
    mabs = Xw.abs.mean(0).view(1, -1)
    out['robuststat'] = _bal_from_imp(W.abs * mabs, sp, beta)
    # (b) BAGGED importance: average |W|*||X|| over 2 disjoint halves of the calib tokens (reduce the
    #     calib-chunk overfit in the survivor ranking -> a more stable/generalizing set)
    h = Xw.shape[0] // 2
    l2a = torch.norm(Xw[:h], dim=0).view(1, -1); l2b = torch.norm(Xw[h:], dim=0).view(1, -1)
    imp_bag = 0.5 * (W.abs * l2a / (l2a.mean + 1e-9) + W.abs * l2b / (l2b.mean + 1e-9))
    out['bagged'] = _bal_from_imp(imp_bag, sp, beta)
    # (c) beta dose variants (screen should say 0.5 ~ optimal)
    out['beta0.4'] = _bal_from_imp(imp, sp, 0.4)
    out['beta0.6'] = _bal_from_imp(imp, sp, 0.6)
    # (d) activation-exponent variants |W|*||X||^a
    out['aexp1.5'] = _bal_from_imp(W.abs * l2.pow(1.5), sp, beta)
    out['aexp0.5'] = _bal_from_imp(W.abs * l2.pow(0.5), sp, beta)
    # (e) STABILITY column stat: mean|X|/std|X| -- keep channels whose activation is STABLE across tokens
    #     (low coeff-of-variation) => transfers across distributions (generalization, not mean-energy)
    stab = (Xw.abs.mean(0) / (Xw.abs.std(0) + 1e-9)).view(1, -1)
    out['stabstat'] = _bal_from_imp(W.abs * l2 * stab.pow(0.5), sp, beta)
    # (f) MIN-MAX robust across 2 calib halves: |W|*min(||X_a||,||X_b||) -- survivors robust to which
    #     sub-distribution activates (a direct min-max generalization criterion on the column energy)
    la = torch.norm(Xw[:h], dim=0).view(1, -1); lb = torch.norm(Xw[h:], dim=0).view(1, -1)
    out['minmax_half'] = _bal_from_imp(W.abs * torch.minimum(la, lb), sp, beta)
    # (g) SET-GEOMETRY / CONDITIONING: up-weight UNIQUE (spanning) columns via 1/[H^-1]_jj (residual
    #     variance of col j given others). Keeps a well-conditioned, spanning survivor set => OBS
    #     reconstructs the pruned mass better OFF-distribution. Categorically != per-entry saliency.
    H_inv = ns.compute_hessian_inverse(Xw, damping=None)
    uniq = (1.0 / (H_inv.diag + 1e-12)).clamp(min=0).view(1, -1)     # unique-variance per column
    uniqg = (uniq / (uniq.mean + 1e-12)).pow(0.25)                   # gentle weight
    uniqs = (uniq / (uniq.mean + 1e-12)).pow(0.5)                    # stronger weight
    out['spanning'] = _bal_from_imp(W.abs * l2 * uniqg, sp, beta)
    out['span_str'] = _bal_from_imp(W.abs * l2 * uniqs, sp, beta)
    # COMBINATIONS of the two consistent screen-winners (softer balance + conditioning)
    out['beta0.35'] = _bal_from_imp(imp, sp, 0.35)
    out['span_b04'] = _bal_from_imp(W.abs * l2 * uniqg, sp, 0.4)
    out['spanstr_b04'] = _bal_from_imp(W.abs * l2 * uniqs, sp, 0.4)
    return out


def main:
    ap = argparse.ArgumentParser
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, default=0.6)
    ap.add_argument("--n-mats", type=int, default=10)
    args = ap.parse_args
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ns_cal = bs.EVAL_CONFIG["n_calibration_samples"]; sl = bs.EVAL_CONFIG["calibration_seq_len"]
    cal_w = bs.get_calibration_data(tok, n_samples=ns_cal, seq_len=sl, dataset_key="wikitext2")
    cal_c = bs.get_calibration_data(tok, n_samples=ns_cal, seq_len=sl, dataset_key="ptb")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV)
    Aw = bs.collect_activations(model, cal_w, DEV)
    Ap = bs.collect_activations(model, cal_c, DEV)
    layers = ns.get_layers(model); paths = bs.get_layer_paths(model)
    items = []
    for li in range(len(layers)):
        for ap_ in paths:
            if Aw.get(f'layer_{li}.{ap_}') is None or Ap.get(f'layer_{li}.{ap_}') is None:
                continue
            mod = layers[li]; ok = True
            for p in ap_.split('.'):
                if not hasattr(mod, p):
                    ok = False; break
                mod = getattr(mod, p)
            if ok and isinstance(mod, nn.Linear):
                items.append((li, ap_, mod))
    step = max(1, len(items) // args.n_mats)
    sample = items[::step][:args.n_mats]
    sp = args.sparsity
    agg = {}
    for (li, ap_, mod) in sample:
        W = mod.weight.data.clone.float.to(DEV)
        Xw = Aw[f'layer_{li}.{ap_}'].float
        Xc = Ap[f'layer_{li}.{ap_}'].float
        if Xw.dim == 3:
            Xw = Xw.reshape(-1, Xw.shape[-1])
        if Xc.dim == 3:
            Xc = Xc.reshape(-1, Xc.shape[-1])
        Xw = Xw[:min(Xw.shape[0], 256)].to(DEV); Xc = Xc[:min(Xc.shape[0], 512)].to(DEV)
        denom = (Xc @ W.t).pow(2).sum.item + 1e-20
        masks = build_masks(W, Xw, sp)
        for nm, mk in masks.items:
            Wc = obs_from_mask(W, Xw, mk)     # OBS fit on WIKI
            D = Wc - W
            err = (Xc @ D.t).pow(2).sum.item / denom   # evaluated on PTB (held-out)
            agg.setdefault(nm, []).append(err)
    import statistics as st
    print(f"# model={args.model} sp={sp} n_mats={len(sample)}  HELD-OUT (ptb) rel recon err, fit on wiki")
    base = st.mean(agg['balanced'])
    for nm in sorted(agg, key=lambda k: st.mean(agg[k])):
        m = st.mean(agg[nm])
        print(f"  {nm:12s} err_ho={m:.5f}  vs balanced {(m/base-1)*100:+.1f}%"
              f"  {'<-- BASE' if nm=='balanced' else ('BETTER' if m<base*0.995 else ('WORSE' if m>base*1.005 else 'tie'))}")


if __name__ == "__main__":
    main
