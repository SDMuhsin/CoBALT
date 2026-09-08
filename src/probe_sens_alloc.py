#!/usr/bin/env python3
"""MEASURE (commission step 1): does GLOBAL-sensitivity-weighted SPARSITY ALLOCATION reduce
held-out global model error vs the deployed UNIFORM per-matrix sparsity?  This is the ONE
in-scope, non-iterative, global-error MASK lever handed off by probe_global_sens (which proved
the non-uniform per-layer sensitivity s_l is INERT for the fixed-bit ENCODING but LIVE for
SPARSITY reallocation = the mask). No method is built yet -- this is a pure go/no-go measurement.

Model of global error (2nd-order, diagonal output-gradient covariance):
    Delta L  ~=  (1/2) Sum_m  (s_m / K_m) * tr(D_m H_m D_m^T)
  s_m  = E|| dLoss/d(output of matrix m) ||^2   (ONE backward pass, dense model, calib)
  K_m  = out_features (so s_m/K_m = per-output-channel mean sensitivity = the isotropic weight)
  D_m  = W_hat_m - W_m   (CoBALT balanced mask + OBS + group-RTN survivor error, at sparsity sp)
  H_m  = X_m^T X_m       (per-matrix calib Hessian)
The per-matrix scalar s_m/K_m governs CROSS-matrix budget allocation (the NEW lever); its within-
matrix per-channel variation is sens_wanda (already lost) and is NOT used here.

Protocol (honest, calib != held-out):
  - fit s_m, e_m(sp), and the ALLOCATION on the calibration activations (H_fit).
  - EVALUATE the resulting global error on a DISJOINT held-out activation set (H_held).
  - Report allocation gain = 1 - G_alloc/G_unif on BOTH H_fit and H_held, WEIGHTED (by s_m/K_m)
    and UNWEIGHTED (w=1, to isolate whether the sensitivity weighting -- not mere error convexity
    -- drives any gain).  Global sparsity is held EXACTLY at TARGET_SP by numel-weighted water-fill.
"""
import argparse, os, sys, math
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from sinq.sparse_quant import quantize_rtn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"
NBITS = 3
GSIZE = 64
BETA = 0.5           # CoBALT column-balance exponent
TARGET_SP = 0.5
SPS = [0.3, 0.4, 0.5, 0.6, 0.7]   # candidate per-matrix sparsities (fixed uniform bits)


def dequant_cobalt(W, X, sp, device):
    """Return dequantized CoBALT weight W_hat (balanced mask beta + OBS + col-norm + group-RTN),
    exactly as nosink deploys it, for a single matrix at sparsity sp."""
    W_comp, mask = ns.balanced_mask_and_obs(W, X, sp, device, col_exp=BETA)
    r, c = ns.compute_norm_scales(W_comp, mask, "col", device)
    W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
    q, scales, zeros, _ = quantize_rtn(W_norm, [0, 2 ** NBITS - 1], group_size=GSIZE)
    if scales.dim == 3:
        scales = scales * r.view(-1, 1, 1)
    else:
        scales = scales * r.view(-1, 1)
    scales = torch.nan_to_num(scales, nan=1e-4, posinf=6e4, neginf=1e-4).clamp(min=1e-8, max=6e4)
    q = q * mask.to(q.dtype)
    K, N = W_comp.shape
    meta = {'sparsity': sp, 'nbits': NBITS, 'requested_nbits': NBITS,
            'group_size': GSIZE, 'shape': (K, N), 'method': 'wanda_obs_rtn_col'}
    W_hat = bs.dequantize_sparse_sinq(q.half, scales.half, zeros.half,
                                      mask.half, c.half, meta).float
    return W_hat


def err(W_hat, W, H):
    """tr(D H D^T), D = W_hat - W, H [N,N]. Normalized by tr(W H W^T) (relative output energy)."""
    D = (W_hat - W).float
    num = torch.einsum('kn,nm,km->', D, H, D)
    den = torch.einsum('kn,nm,km->', W, H, W).clamp(min=1e-30)
    return float(num / den), float(num)


def gram(Xdict, key, device):
    X = Xdict[key].float.to(device)
    if X.dim == 3:
        X = X.reshape(-1, X.shape[-1])
    return X.t @ X


def water_fill(e_by_sp, weights, numel, target_sp, sps):
    """Choose per-matrix sp_m from the grid to minimize Sum_m weights_m * e_m(sp_m) subject to
    Sum_m numel_m*sp_m = target_sp*Sum numel_m. Lagrangian: sp_m = argmin_s w_m e_m(s)+lam*numel_m*s;
    bisect lam to hit the budget. Returns (chosen_sp[list], global weighted error at chosen)."""
    M = len(numel)
    budget = target_sp * sum(numel)

    def alloc(lam):
        # minimize Sum w_m e_m(s) s.t. Sum numel_m s >= budget. Lagrangian per matrix:
        #   minimize  w_m e_m(s) - lam * numel_m * s   (lam>=0 REWARDS more sparsity).
        # error e(s) increases with s; larger lam -> pick larger s -> more sparsity used.
        chosen = []
        for m in range(M):
            best_s, best_v = sps[0], float('inf')
            for si, s in enumerate(sps):
                v = weights[m] * e_by_sp[m][si] - lam * numel[m] * s
                if v < best_v:
                    best_v, best_s = v, s
            chosen.append(best_s)
        return chosen

    lo, hi = 0.0, 1e18
    # used(lam) is INCREASING in lam; bisect for used ~= budget (equality when feasible).
    for _ in range(200):
        mid = (lo + hi) / 2
        used = sum(numel[m] * alloc(mid)[m] for m in range(M))
        if used < budget:
            lo = mid
        else:
            hi = mid
    chosen = alloc(hi)
    return chosen


def run(MODEL, n_calib):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16,
                                                 device_map=DEV, low_cpu_mem_usage=True)
    model.eval
    layers = bs.get_transformer_layers(model)
    paths = bs.get_layer_paths(model)

    # ---- one backward pass: per-matrix output-gradient sensitivity s_m ----
    import torch.nn as nn
    targets = {}   # key -> module
    for li, layer in enumerate(layers):
        for p in paths:
            mod = layer
            ok = True
            for part in p.split('.'):
                if not hasattr(mod, part):
                    ok = False; break
                mod = getattr(mod, part)
            if ok and isinstance(mod, nn.Linear):
                targets[f'layer_{li}.{p}'] = mod
    caps = {}
    hooks = []
    def mk(k):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o.retain_grad; caps[k] = o
        return h
    for k, mod in targets.items:
        hooks.append(mod.register_forward_hook(mk(k)))
    sens_batch = bs.get_calibration_data(tok, n_samples=1, seq_len=256, dataset_key="wikitext2").to(DEV)
    out = model(sens_batch, labels=sens_batch)
    out.loss.backward
    s_m = {}
    for k, o in caps.items:
        g = o.grad
        if g is None:
            s_m[k] = 0.0; continue
        g = g.reshape(-1, g.shape[-1]).float
        s_m[k] = float((g * g).sum(-1).mean)   # E||grad||^2 over tokens
    for hk in hooks:
        hk.remove
    model.zero_grad(set_to_none=True)
    del caps, out, sens_batch
    torch.cuda.empty_cache

    # ---- activations: fit (calib) + disjoint held-out ----
    torch.manual_seed(1)
    cal_fit = bs.get_calibration_data(tok, n_samples=n_calib, seq_len=256, dataset_key="wikitext2")
    torch.manual_seed(777)
    cal_held = bs.get_calibration_data(tok, n_samples=n_calib, seq_len=256, dataset_key="ptb")
    Xfit = bs.collect_activations(model, cal_fit, DEV)
    Xheld = bs.collect_activations(model, cal_held, DEV)
    for _l in ns.get_layers(model):
        _l.to("cpu")
    torch.cuda.empty_cache

    # ---- per-matrix e_m(sp) on H_fit and H_held ----
    rows = []  # (key, K, numel, s_over_K, [e_fit per sp], [e_held per sp])
    for li, layer in enumerate(bs.get_transformer_layers(model)):
        layer = layer.to(DEV)
        for p in paths:
            mod = layer
            ok = True
            for part in p.split('.'):
                if not hasattr(mod, part):
                    ok = False; break
                mod = getattr(mod, part)
            if not (ok and isinstance(mod, nn.Linear)):
                continue
            key = f'layer_{li}.{p}'
            if key not in Xfit or key not in Xheld:
                continue
            W = mod.weight.data.float.to(DEV)
            K, N = W.shape
            Hf = gram(Xfit, key, DEV)
            Hh = gram(Xheld, key, DEV)
            ef, eh = [], []
            for sp in SPS:
                W_hat = dequant_cobalt(W, Xfit[key].to(DEV), sp, DEV)
                rf, _ = err(W_hat, W, Hf)
                rh, _ = err(W_hat, W, Hh)
                ef.append(rf); eh.append(rh)
            rows.append((key, K, W.numel, s_m.get(key, 0.0) / max(K, 1), ef, eh))
            del W, Hf, Hh
            torch.cuda.empty_cache
        layer.to("cpu"); torch.cuda.empty_cache

    # ---- allocation experiment ----
    keys = [r[0] for r in rows]
    numel = [r[2] for r in rows]
    wsens = [r[3] for r in rows]
    E_fit = [r[4] for r in rows]
    E_held = [r[5] for r in rows]
    ui = SPS.index(TARGET_SP)

    def global_err(chosen_idx, Etab, weights):
        return sum(weights[m] * Etab[m][chosen_idx[m]] for m in range(len(rows)))

    unif_idx = [ui] * len(rows)
    # SHARED average error curve phi(sp) (mean over matrices, normalized) -> used by the s_m-ONLY
    # closed-form allocation: replaces each matrix's own e_m(sp) with the universal phi, so ONLY
    # s_m and numel differentiate the allocation (no per-matrix error shape = clean global-sens mask).
    import statistics
    phi = [statistics.fmean(E_fit[m][si] for m in range(len(rows))) for si in range(len(SPS))]
    E_phi = [phi for _ in range(len(rows))]

    results = {}
    arms = [("sens", wsens, E_fit), ("flat", [1.0] * len(rows), E_fit),
            ("sensonly", wsens, E_phi)]  # sensonly: sens weights, shared phi curve
    for wname, weights, Efit_used in arms:
        chosen = water_fill(Efit_used, weights, numel, TARGET_SP, SPS)  # FIT on calib
        cidx = [SPS.index(s) for s in chosen]
        realized_sp = sum(numel[m] * chosen[m] for m in range(len(rows))) / sum(numel)
        # EVALUATE the chosen allocation on the TRUE per-matrix error (sens-weighted), fit & held
        for hname, Etab in [("fit", E_fit), ("held", E_held)]:
            gu = global_err(unif_idx, Etab, wsens)   # eval always vs true global (sens-weighted) err
            ga = global_err(cidx, Etab, wsens)
            results[(wname, hname)] = (1.0 - ga / gu, realized_sp)
        results[(wname, "profile")] = chosen

    print(f"\n===== {MODEL} (loss-sens matrices={len(rows)}, target_sp={TARGET_SP}, nbits={NBITS}) =====")
    print("  [all arms EVALUATED on the SAME true global (sens-weighted) held-out error; lower=better]")
    for wname in ["sens", "flat", "sensonly"]:
        gf, rsp = results[(wname, "fit")]
        gh, _ = results[(wname, "held")]
        print(f"  alloc={wname:9s}  reduce-global-err FIT={gf*100:+6.2f}%  HELD={gh*100:+6.2f}%  (realized_sp={rsp:.4f})")
    gs = results[("sens", "held")][0]; gf = results[("flat", "held")][0]; go = results[("sensonly", "held")][0]
    print(f"  >>> HELD-OUT global-err reduction: full-water-fill {gs*100:+.2f}%  |  s_m-ONLY closed-form {go*100:+.2f}%"
          f"  |  flat(no-sens) {gf*100:+.2f}%")
    print(f"  >>> s_m-only captures {go/gs*100:.0f}% of full water-fill; sensitivity vs flat = {(gs-gf)*100:+.2f}%")
    # allocation profile: early vs late (does it protect early layers as sens suggests?)
    prof = results[("sens", "profile")]
    early = [prof[m] for m in range(len(rows)) if int(keys[m].split('.')[0].split('_')[1]) < len(bs.get_transformer_layers(model)) // 2]
    late = [prof[m] for m in range(len(rows)) if int(keys[m].split('.')[0].split('_')[1]) >= len(bs.get_transformer_layers(model)) // 2]
    print(f"  profile: early-half mean sp={sum(early)/len(early):.3f}  late-half mean sp={sum(late)/len(late):.3f}")
    return results


def main:
    ap = argparse.ArgumentParser
    ap.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    ap.add_argument("--n-calib", type=int, default=16)
    args = ap.parse_args
    for m in args.models.split(","):
        try:
            run(m.strip, args.n_calib)
        except Exception as e:
            import traceback; traceback.print_exc
            print(f"[{m}] FAILED: {e}")


if __name__ == "__main__":
    main
