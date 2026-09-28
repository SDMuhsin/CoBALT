#!/usr/bin/env python3
"""GROUPING axis (AXIS F) precondition, attempt-8d. Deployed groups = 128 CONTIGUOUS columns; the
per-(row,group) RTN scale = the group's widest per-column range. A single GLOBAL column permutation
(shared across K rows, bpw ~ log2(N)/K ~= 0.005, negligible) that groups columns with SIMILAR
per-column magnitude could tighten low-range groups. Precondition (weight-only, no Hessian): does
grouping columns SORTED by a shared per-column statistic reduce total group-RTN weight-MSE vs
contiguous? Is the reduction balance-favoring (balanced >> wanda)? c already per-column-normalizes,
so the only exploitable residual is per-column tail/max structure SHARED across rows. If ratio~=1,
AXIS F dead. sort keys: colmax = max_row |W_norm[:,j]| ; colkurt = per-column kurtosis over rows.
"""
import argparse, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from sinq.sparse_quant import quantize_rtn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
from probe_heldout_gap import collect  # noqa
from probe_sscale import magnitude_mask_and_obs  # noqa

SP, BETA, NBITS, GROUP = 0.5, 0.5, 3, 128


def group_rtn_mse(W_norm, mask, gsize):
    """pooled survivor weight-MSE of deployed group-RTN over the given column ORDER (already applied)."""
    q, s, z, _ = quantize_rtn(W_norm, [0, 2 ** NBITS - 1], group_size=gsize)
    from eout_quant import dequant_deployed
    Wt = dequant_deployed(q, s, z, torch.ones_like(W_norm),
                          torch.ones(W_norm.shape[1], device=W_norm.device))
    d = (Wt - W_norm) * mask.float()
    return float((d * d).sum().item())


def run_model(MODEL, device="cuda"):
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    A = collect(model, tok, device, "wikitext2", 16)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache()
    lp = bs.get_layer_paths(model); layers = ns.get_layers(model); nl = len(layers)
    sample = sorted(set([1, nl // 2, nl - 2]))
    print(f"\n######## {MODEL} sample={sample} g={GROUP} ########", flush=True)
    MK = ["balanced", "wanda", "magnitude"]
    acc = {mk: {"contig": 0.0, "colmax": 0.0, "colkurt": 0.0} for mk in MK}
    nmat = 0
    for li in sample:
        layer = layers[li].to(device)
        for ap in lp:
            parts = ap.split('.'); parent = layer; ok = True
            for p in parts[:-1]:
                if not hasattr(parent, p): ok = False; break
                parent = getattr(parent, p)
            if not ok or not hasattr(parent, parts[-1]): continue
            lin = getattr(parent, parts[-1])
            if not isinstance(lin, torch.nn.Linear): continue
            key = f'layer_{li}.{ap}'
            if A.get(key) is None: continue
            W = lin.weight.data.clone().float().to(device); K, N = W.shape
            block = bs._largest_divisor_leq(N, GROUP)
            if not (N > block and N % block == 0): continue
            for mk in MK:
                if mk == "balanced":
                    W_comp, mask = ns.balanced_mask_and_obs(W, A[key].to(device), SP, device, col_exp=BETA)
                elif mk == "wanda":
                    W_comp, mask = ns.wanda_mask_and_obs(W, A[key].to(device), SP, device, scope='per_row')
                else:
                    W_comp, mask = magnitude_mask_and_obs(W, A[key].to(device), SP, device)
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device)
                W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
                mkf = mask.float()
                acc[mk]["contig"] += group_rtn_mse(W_norm, mask, block)
                # colmax sort
                colmax = W_norm.abs().amax(0)
                perm = torch.argsort(colmax)
                acc[mk]["colmax"] += group_rtn_mse(W_norm[:, perm], mask[:, perm], block)
                # colkurt sort (per-column 4th moment over rows, survivors)
                cnt = mkf.sum(0).clamp(min=1.0)
                mu = (W_norm * mkf).sum(0) / cnt
                d2 = ((W_norm - mu.view(1, -1)) ** 2 * mkf)
                var = d2.sum(0) / cnt
                d4 = ((W_norm - mu.view(1, -1)) ** 4 * mkf).sum(0) / cnt
                kurt = (d4 / var.clamp(min=1e-12) ** 2)
                perm2 = torch.argsort(kurt)
                acc[mk]["colkurt"] += group_rtn_mse(W_norm[:, perm2], mask[:, perm2], block)
            nmat += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache()
    print(f"  pooled {nmat} matrices/mask. group-RTN weight-MSE ratio vs contiguous (<1 => grouping helps).", flush=True)
    for mk in MK:
        base = acc[mk]["contig"]
        print(f"    {mk:<10} colmax={acc[mk]['colmax']/base:.4f}  colkurt={acc[mk]['colkurt']/base:.4f}", flush=True)


def main():
    P = argparse.ArgumentParser(); P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args()
    for m in Aa.models.split(","): run_model(m.strip())


if __name__ == "__main__":
    main()
