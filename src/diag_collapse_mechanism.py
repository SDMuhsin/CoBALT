#!/usr/bin/env python3
"""CROSS-MODEL mechanism probe: WHY does uniform per-row Wanda PRUNING (fp16 survivors, no quant)
catastrophically collapse on gemma-2b (wiki sp0.7 = 5.9e10) but degrade gracefully on Llama-3.2-3B
/ Qwen2.5-3B / StarCoder2-3B (125/139/351)? The collapse is the enabling event for CoBALT's win, so
find the tensor-level property P that (a) is a gemma outlier, (b) explains the pruning collapse,
(c) is what CoBALT's column balance corrects.

Per matrix, under the UNIFORM per-row Wanda mask at sparsity SP (the collapsing baseline):
  col_dead%   = % of input columns kept in <5% of output rows (starved input channels)
  col_CV      = CV of per-column keep fraction (column imbalance of the survivor set)
  actmax/med  = per-column ||X|| max / median  (activation-outlier severity)
  act_top.1%  = fraction of sum ||X_j||^2 in the top-0.1% columns (outlier mass concentration)
  imp_gini    = Gini of per-column total importance |W|.||X|| (mass concentration)
and the balanced mask (col_exp=0.5, CoBALT's lever) col_dead% / col_CV (leg c: should drop).

Aggregated (mean over matrices) by matrix TYPE and overall. Rank confirms the mechanism only;
the verdict is the end-to-end grids already run (rule #2)."""
import os, sys, argparse
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
import nosink as ns  # noqa

DEV = "cuda"


def _thr_mask(imp, sparsity, scope):
    K, N = imp.shape
    if scope == 'per_row':
        kp = int(N * sparsity)
        thr = torch.kthvalue(imp, kp, dim=1, keepdim=True).values
        return (imp > thr).float()
    n_prune = int(K * N * sparsity)
    flat = imp.view(-1)
    thr = torch.kthvalue(flat, n_prune).values
    return (flat > thr).view(K, N).float()


def wanda_mask(W, act_norms, sp):
    return _thr_mask(W.abs() * act_norms.view(1, -1), sp, 'per_row')


def balanced_mask(W, act_norms, sp, col_exp):
    K, N = W.shape
    imp = W.abs() * act_norms.view(1, -1)
    kr, kc = int(N * sp), int(K * sp)
    if kr > 0:
        qr = torch.kthvalue(imp, kr, dim=1, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qr
    if col_exp > 0 and kc > 0:
        qc = torch.kthvalue(imp, kc, dim=0, keepdim=True).values.clamp(min=1e-30)
        imp = imp / qc.pow(col_exp)
    return _thr_mask(imp, sp, 'global')


def colstats(mask):
    kf = mask.mean(0)                                  # per-column keep fraction [N]
    cv = (kf.std() / (kf.mean() + 1e-12)).item()
    dead = (kf < 0.05).float().mean().item() * 100
    return cv, dead


def act_stats(X):
    # X: [tokens, N]
    n = torch.norm(X.float(), dim=0)                   # per-column L2 norm [N]
    med = n.median().clamp(min=1e-20)
    mx = n.max()
    e = n.pow(2)                                        # energy per column
    k = max(1, int(round(0.001 * n.numel())))
    top = torch.topk(e, k).values.sum() / e.sum().clamp(min=1e-20)
    return (mx / med).item(), top.item()


def gini(v):
    v = v.float().flatten().sort().values
    n = v.numel()
    if n == 0 or v.sum() <= 0:
        return 0.0
    idx = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    return ((2 * idx - n - 1) * v).sum().item() / (n * v.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, default=0.70)
    ap.add_argument("--col-exp", type=float, default=0.5)
    ap.add_argument("--layer-stride", type=int, default=2, help="sample every Nth layer to save time")
    args = ap.parse_args()

    SP = args.sparsity
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, DEV)
    acts = bs.collect_activations(model, cal, DEV)
    layers = ns.get_layers(model)
    paths = bs.get_layer_paths(model)
    print(f"# model={args.model} sp={SP} col_exp={args.col_exp} n_layers={len(layers)} paths={paths}", flush=True)

    rows = []  # (type, wcv, wdead, amaxmed, atop, igini, bcv, bdead)
    for li in range(0, len(layers), args.layer_stride):
        layer = layers[li].to(DEV)
        for ap_ in paths:
            mod = layer
            ok = True
            for p in ap_.split('.'):
                if not hasattr(mod, p):
                    ok = False; break
                mod = getattr(mod, p)
            if not ok or not isinstance(mod, nn.Linear):
                continue
            X = acts.get(f'layer_{li}.{ap_}')
            if X is None:
                continue
            W = mod.weight.data.clone()
            Xd = X.to(DEV)
            if Xd.dim() == 3:
                Xd = Xd.reshape(-1, Xd.shape[-1])
            Xd = Xd[:min(Xd.shape[0], 256)]
            W = W.float()
            anorm = torch.norm(Xd.float(), dim=0)          # [N]
            m_w = wanda_mask(W, anorm, SP)
            m_b = balanced_mask(W, anorm, SP, args.col_exp)
            wcv, wdead = colstats(m_w)
            bcv, bdead = colstats(m_b)
            amm, atop = act_stats(Xd)
            imp_col = (W.abs() * anorm.view(1, -1)).sum(0)
            ig = gini(imp_col)
            t = ap_.split('.')[-1]
            rows.append((t, wcv, wdead, amm, atop, ig, bcv, bdead))
        layers[li].to("cpu")
        del layer
        torch.cuda.empty_cache()

    # aggregate
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r[1:])
    hdr = f"{'type':10s} {'n':>3s} | {'W_colCV':>8s} {'W_dead%':>8s} | {'aMax/Med':>9s} {'aTop.1%':>8s} {'impGini':>8s} | {'B_colCV':>8s} {'B_dead%':>8s}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    def line(tag, vs):
        import statistics as st
        m = [st.mean([v[i] for v in vs]) for i in range(7)]
        print(f"{tag:10s} {len(vs):3d} | {m[0]:8.2f} {m[1]:8.1f} | {m[2]:9.1f} {m[3]:8.3f} {m[4]:8.3f} | {m[5]:8.2f} {m[6]:8.1f}", flush=True)
    for t in by:
        line(t, by[t])
    line("ALL", [r[1:] for r in rows])


if __name__ == "__main__":
    main()
