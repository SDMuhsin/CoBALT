#!/usr/bin/env python3
"""MEASURE (attempt-10, lever #6, step 1): at the 3-bit sp0.5 operating point, does a LOSS/GRADIENT-aware
WITHIN-matrix column-importance signal pick a DIFFERENT survivor set than CoBALT-balanced / Wanda? This
is the cheap information filter BEFORE building anything (calib tr(DHD) mispredicts downstream -> do NOT
screen on it; screen on whether the signal even CARRIES NEW INFORMATION vs the deployed mask).

Collapse-safe by construction: fixed uniform sp0.5 (no budget reallocation -- that axis is a measured
downstream negative, results/sens_alloc/FINDINGS.md). We only change WHICH survivors are kept.

Signals compared (per matrix, at sp0.5), all masks built with the SAME balanced row+col quantile machinery
(so we isolate the IMPORTANCE signal, not the thresholding):
  base   : CoBALT-balanced importance   = |W|.||X||         (col_exp=0.5)                [deployed]
  wanda  : |W|.||X||  per-row threshold (saliency, no balance)
  gcol   : |W|.||X|| .|g|_col^a  -- tilt columns by the OUTPUT-GRADIENT column magnitude
           |g|_col_j = sqrt(E_n[ (sum_i |g_i(n) w_ij|)^2 ]) proxy: use per-col ||g-weighted W . x||.
           Simplify to a per-column scalar gw_j = sqrt(E[g_i^2]) aggregated: gw_j = || (|W|^T @ sqrt(s_row)) *||X||_j
           i.e. weight Wanda column importance by how much its rows matter to the loss.
  wonly  : |W| only (no activation) -- control

Report per model: mean Jaccard(base, X) over matrices (1.0 = identical survivor set = redundant), and the
rank-correlation of the per-column importance vectors vs Wanda ||X||. HIGH overlap => dead before build.
LOW overlap => a real within-matrix lever to SMOKE downstream next."""
import argparse, os, sys
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"
SP = 0.5
BETA = 0.5


def balanced_keepmask(imp, sparsity):
    """CoBALT balanced thresholding on a given importance map imp[K,N] (row quantile + col^beta quantile
    + global top-k), matching nosink.balanced_mask_and_obs' mask branch."""
    K, N = imp.shape
    kr, kc = int(N * sparsity), int(K * sparsity)
    imp = imp.clone
    if kr > 0:
        qr = torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qr
    if kc > 0:
        qc = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qc.pow(BETA)
    n_prune = int(K * N * sparsity)
    thr = torch.kthvalue(imp.view(-1), n_prune).values
    return (imp.view(-1) > thr).view(K, N)


def jaccard(a, b):
    a = a.bool; b = b.bool
    inter = (a & b).sum.float
    union = (a | b).sum.float.clamp(min=1)
    return float(inter / union)


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

    # per-output-channel loss sensitivity s_i (backward pass), per matrix
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
    caps = {}; cap_in = {}; hooks = []
    def mk(k):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o.retain_grad; caps[k] = o
            cap_in[k] = (inp[0] if isinstance(inp, tuple) else inp).detach
        return h
    for k, mod in targets.items:
        hooks.append(mod.register_forward_hook(mk(k)))
    si = {k: None for k in targets}
    fint = {k: None for k in targets}   # exact empirical Fisher interaction E[g_i^2 x_j^2]  [K,N]
    cnt = 0
    torch.manual_seed(0)
    data = bs.get_calibration_data(tok, n_samples=4, seq_len=256, dataset_key="wikitext2")
    for i in range(data.shape[0]):
        model.zero_grad(set_to_none=True); caps.clear; cap_in.clear
        out = model(data[i:i+1].to(DEV), labels=data[i:i+1].to(DEV)); out.loss.backward
        for k, o in caps.items:
            if o.grad is None:
                continue
            g = o.grad.reshape(-1, o.grad.shape[-1]).float       # [n,K]
            si[k] = (g * g).mean(0) if si[k] is None else si[k] + (g * g).mean(0)
            x = cap_in[k].reshape(-1, cap_in[k].shape[-1]).float  # [n,N]
            # E[g_i^2 x_j^2] = (g^2)^T (x^2) / n  -> [K,N]; cap token count to bound memory
            m = min(g.shape[0], 512)
            fi = (g[:m] ** 2).t @ (x[:m] ** 2) / m
            fint[k] = fi if fint[k] is None else fint[k] + fi
        cnt += 1
    for hk in hooks:
        hk.remove
    model.zero_grad(set_to_none=True)
    torch.manual_seed(1)
    cal = bs.get_calibration_data(tok, n_samples=n_calib, seq_len=256, dataset_key="wikitext2")
    Xd = bs.collect_activations(model, cal, DEV)
    for _l in ns.get_layers(model):
        _l.to("cpu")
    torch.cuda.empty_cache

    agg = {"wanda": [], "gcol": [], "wonly": [], "fint": []}
    colcorr = {"gcol": []}
    for li, layer in enumerate(bs.get_transformer_layers(model)):
        layer = layer.to(DEV)
        for p in paths:
            mod = layer; ok = True
            for part in p.split('.'):
                if not hasattr(mod, part):
                    ok = False; break
                mod = getattr(mod, part)
            if not (ok and isinstance(mod, nn.Linear)):
                continue
            key = f'layer_{li}.{p}'
            if key not in Xd or si[key] is None:
                continue
            W = mod.weight.data.float.to(DEV)
            X = Xd[key].float.to(DEV)
            if X.dim == 3:
                X = X.reshape(-1, X.shape[-1])
            X = X[:min(X.shape[0], 256)]
            xn = torch.norm(X, dim=0)                     # ||X||_j  [N]
            s = si[key].to(DEV).clamp(min=0)              # [K] E[g_i^2]
            base_imp = W.abs * xn.view(1, -1)           # |W|.||X||
            m_base = balanced_keepmask(base_imp, SP)
            # wanda per-row (saliency, no col balance): here reuse balanced machinery w/ BETA=0 => rows-only
            kN = int(base_imp.shape[1] * SP)
            impw = base_imp / torch.kthvalue(base_imp, kN, dim=1, keepdim=True).values.clamp(min=1e-30)
            thr = torch.kthvalue(impw.reshape(-1), int(base_imp.numel * SP)).values
            m_wanda = (impw.reshape(-1) > thr).view_as(base_imp)
            # gcol: tilt columns by loss-relevance of their rows: gw_j = || sqrt(s) * |W| ||_col weighted by ||X||
            gW = (W.abs * s.sqrt.view(-1, 1))         # rows weighted by sqrt(E[g^2])
            gcol_j = gW.sum(0) * xn                       # [N] loss-aware column importance
            imp_g = (W.abs * s.sqrt.view(-1, 1)) * xn.view(1, -1)  # per-entry loss-aware
            m_gcol = balanced_keepmask(imp_g, SP)
            m_wonly = balanced_keepmask(W.abs, SP)
            # fint: exact joint-Fisher INTERACTION importance sqrt(E[g_i^2 x_j^2])*|W| (NON-separable part
            # is the only thing that can move the balanced mask, per the GCOL row-absorption result).
            imp_fint = fint[key].to(DEV).clamp(min=0).sqrt * W.abs
            m_fint = balanced_keepmask(imp_fint, SP)
            agg["wanda"].append(jaccard(m_base, m_wanda))
            agg["gcol"].append(jaccard(m_base, m_gcol))
            agg["wonly"].append(jaccard(m_base, m_wonly))
            agg["fint"].append(jaccard(m_base, m_fint))
            # column-importance rank corr: gcol_j vs wanda col imp (|W|.||X|| summed over rows)
            wcol_j = base_imp.sum(0)
            a = gcol_j.argsort.argsort.float; b = wcol_j.argsort.argsort.float
            colcorr["gcol"].append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
            del W, X
        layer.to("cpu"); torch.cuda.empty_cache

    def mean(x):
        return sum(x) / len(x) if x else float('nan')
    print(f"\n===== {MODEL} (matrices={len(agg['wanda'])}, sp={SP}) =====")
    print(f"  Jaccard(balanced, wanda-per-row) = {mean(agg['wanda']):.3f}   (deployed base vs saliency)")
    print(f"  Jaccard(balanced, GCOL loss-aware)= {mean(agg['gcol']):.3f}   (NEW lever survivor overlap)")
    print(f"  Jaccard(balanced, |W|-only)      = {mean(agg['wonly']):.3f}   (control)")
    print(f"  Jaccard(balanced, FINT interaction)={mean(agg['fint']):.3f}   (joint E[g^2 x^2] NON-separable)")
    print(f"  col-imp rank-corr GCOL vs Wanda  = {mean(colcorr['gcol']):.3f}   (1.0 => no new column info)")
    for nm in ("gcol", "fint"):
        ov = mean(agg[nm])
        print(f"  >>> {nm.upper} vs balanced: {(1-ov)*100:.1f}% of survivors DIFFER "
              f"=> {'REDUNDANT (skip build)' if ov > 0.9 else 'DIFFERENT (worth a downstream smoke)'}")


def main:
    ap = argparse.ArgumentParser
    ap.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    ap.add_argument("--n-calib", type=int, default=16)
    args = ap.parse_args
    for m in args.models.split(","):
        try:
            run(m.strip, args.n_calib)
        except Exception as e:
            import traceback; traceback.print_exc; print(f"[{m}] FAILED: {e}")


if __name__ == "__main__":
    main
