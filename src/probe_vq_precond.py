#!/usr/bin/env python3
"""VECTOR-QUANTIZATION / joint-structure precondition (attempt-8g, keep-grinding). Scalar RTN is
optimal per-position on a UNIFORM marginal, but if survivor PAIRS carry joint CORRELATION, a vector
quantizer (or a fixed rotation) could beat scalar at matched rate. Cheap weight-only test: mean
|Pearson correlation| between survivor columns within a group (over rows), and mean |corr| to the
NEAREST neighbour column. If ~0 => survivors jointly independent => VQ/rotation gain ~ space-filling
only (universal ~0.17 bit), dead-for-CoBALT-specificity. If high & balance-favoring => real lead.
balanced vs wanda vs magnitude, 3 families. c-normalized survivors (deployed frame).
"""
import argparse, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
from probe_heldout_gap import collect  # noqa
from probe_sscale import magnitude_mask_and_obs  # noqa

SP, BETA, GROUP = 0.5, 0.5, 128


def mean_abs_adjcorr(W_norm, mask, gsize):
    """mean |corr(col_j, col_{j+1})| within groups, over survivor rows (pairwise-complete)."""
    K, N = W_norm.shape
    if not (N > gsize and N % gsize == 0):
        return None
    m = mask.bool()
    # standardize columns over survivors
    cnt = m.float().sum(0).clamp(min=2.0)
    mu = torch.where(m, W_norm, torch.zeros_like(W_norm)).sum(0) / cnt
    Wc = torch.where(m, W_norm - mu.view(1, -1), torch.zeros_like(W_norm))
    sd = (torch.where(m, Wc * Wc, torch.zeros_like(Wc)).sum(0) / cnt).sqrt().clamp(min=1e-8)
    Z = Wc / sd.view(1, -1)                       # [K,N] standardized, 0 at pruned
    # adjacent within-group pairs: j and j+1 not crossing a group boundary
    cols = torch.arange(N, device=W_norm.device)
    same_group = (cols[:-1] // gsize) == (cols[1:] // gsize)
    a = Z[:, :-1]; b = Z[:, 1:]; both = (m[:, :-1] & m[:, 1:]).float()
    n_both = both.sum(0).clamp(min=1.0)
    corr = (a * b * both).sum(0) / n_both          # E[za zb] over jointly-surviving rows ~ corr
    corr = corr[same_group]
    return float(corr.abs().mean().item())


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
    acc = {mk: [0.0, 0] for mk in MK}
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
            for mk in MK:
                if mk == "balanced":
                    W_comp, mask = ns.balanced_mask_and_obs(W, A[key].to(device), SP, device, col_exp=BETA)
                elif mk == "wanda":
                    W_comp, mask = ns.wanda_mask_and_obs(W, A[key].to(device), SP, device, scope='per_row')
                else:
                    W_comp, mask = magnitude_mask_and_obs(W, A[key].to(device), SP, device)
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device)
                W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
                v = mean_abs_adjcorr(W_norm, mask, block)
                if v is not None:
                    acc[mk][0] += v; acc[mk][1] += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache()
    print(f"  mean |adjacent within-group survivor correlation| (0 => jointly independent => VQ dead):", flush=True)
    for mk in MK:
        print(f"    {mk:<10} mean|corr|={acc[mk][0]/max(1,acc[mk][1]):.4f}", flush=True)


def main():
    P = argparse.ArgumentParser(); P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args()
    for m in Aa.models.split(","): run_model(m.strip())


if __name__ == "__main__":
    main()
