#!/usr/bin/env python3
"""MEASURE-FIRST for the exact-doubly-balanced candidate. Greedy degree-capped support: sort entries by
raw wanda importance |W|.||X|| desc, keep if BOTH its row and column are below their exact keep-caps
kr=round((1-sp)*N), kc=round((1-sp)*K); then a fill pass tops up under-filled ROWS to hit exact global
sparsity. Enforces EXACT per-row + (near-)exact per-column keep-rates in ONE non-iterative pass (unlike
soft-beta's global-top-k, which only APPROXIMATES double balance). Question: does this survivor set
DIFFER from CoBALT's soft-beta set (Jaccard), or is it absorbed like pure row/col tilts? If Jaccard is
high (~>0.92) it's absorbed -> skip. If meaningfully different, it's a real lever -> smoke."""
import os, sys, argparse
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
DEV = "cuda"


def exact_doubly_balanced(imp, sp):
    """Greedy degree-capped doubly-balanced keep-mask [K,N] (float). One-shot."""
    K, N = imp.shape
    kr = int(round((1.0 - sp) * N))   # survivors per row
    kc = int(round((1.0 - sp) * K))   # survivors per column
    if kr <= 0:
        return torch.zeros_like(imp)
    dev = imp.device
    order = torch.argsort(imp.reshape(-1), descending=True)   # global desc
    rows = (order // N); cols = (order % N)
    row_cnt = torch.zeros(K, dtype=torch.int32, device=dev)
    col_cnt = torch.zeros(N, dtype=torch.int32, device=dev)
    keep = torch.zeros(K * N, dtype=torch.bool, device=dev)
    # PASS 1: greedy with both caps (vectorization-hard due to sequential caps -> chunked python loop
    # over the sorted order, but only until all rows are full). Move to cpu ints for speed.
    ro = rows.to('cpu').numpy(); co = cols.to('cpu').numpy(); od = order.to('cpu').numpy()
    rc = [0] * K; cc = [0] * N
    keepl = bytearray(K * N)
    filled_rows = 0
    for t in range(len(od)):
        i = int(ro[t]); j = int(co[t])
        if rc[i] < kr and cc[j] < kc:
            keepl[od[t]] = 1; rc[i] += 1; cc[j] += 1
            if rc[i] == kr:
                filled_rows += 1
    # PASS 2: fill under-filled rows (blocked by col caps) with their best remaining entries, col-cap
    # relaxed, to reach EXACT per-row kr (=> exact global sparsity).
    for t in range(len(od)):
        i = int(ro[t])
        if rc[i] < kr and keepl[od[t]] == 0:
            keepl[od[t]] = 1; rc[i] += 1
    keep = torch.tensor(list(keepl), dtype=torch.float32, device=dev).view(K, N)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, default=0.6)
    ap.add_argument("--n-mats", type=int, default=8)
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
    sp = args.sparsity
    jacc = []; colcv_b = []; colcv_e = []
    for (li, ap_, mod, X) in sample:
        W = mod.weight.data.clone().float().to(DEV)
        Xd = X.to(DEV)
        if Xd.dim() == 3:
            Xd = Xd.reshape(-1, Xd.shape[-1])
        Xd = Xd[:min(Xd.shape[0], 256)]
        _, mb = ns.balanced_mask_and_obs(W, Xd, sp, DEV, col_exp=0.5, no_obs=True)
        imp = W.abs() * torch.norm(Xd, dim=0).view(1, -1)
        me = exact_doubly_balanced(imp, sp)
        inter = (mb.bool() & me.bool()).sum().item()
        union = (mb.bool() | me.bool()).sum().item()
        jacc.append(inter / max(union, 1))
        # column keep-rate CV (lower = more balanced)
        cb = mb.sum(0); ce = me.sum(0)
        colcv_b.append((cb.std() / (cb.mean() + 1e-9)).item())
        colcv_e.append((ce.std() / (ce.mean() + 1e-9)).item())
    import statistics as st
    print(f"# model={args.model} sp={sp} n={len(sample)}")
    print(f"Jaccard(balanced, exact_doubly) = {st.mean(jacc):.4f}  (min {min(jacc):.4f} max {max(jacc):.4f})")
    print(f"col keep-rate CV: balanced={st.mean(colcv_b):.4f}  exact_doubly={st.mean(colcv_e):.4f} "
          f"(exact should be <= balanced if it balances columns harder)")


if __name__ == "__main__":
    main()
