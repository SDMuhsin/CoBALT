#!/usr/bin/env python3
"""SURVIVOR DISTRIBUTION vs SPARSITY REGIME (attempt-8h). The i.i.d.-uniform floor (=> RTN optimal
for values) was measured at ONE operating point: sp0.5. But CoBALT's edge PEAKS in the HIGH-SPARSITY
COLLAPSE regime (sp0.7-0.8, [[cobalt-win-is-bottleneck-contingent]]), where OBS redistributes far more
pruned mass into fewer survivors -- which could break uniformity/independence EXACTLY where CoBALT wins,
reopening the value-encoding axis. Sweep sp in {0.4..0.8}: per-group-affine-normalized survivor KURTOSIS
(uniform=1.8, gauss=3.0, heavy-tail>3) + mean |adjacent within-group correlation| (independent~0).
balanced (CoBALT) mask, 3 families. If kurtosis departs from 1.8 or corr rises at high sp => value
encoding REOPENS in the winning regime.
"""
import argparse, os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
from probe_heldout_gap import collect  # noqa
from probe_vq_precond import mean_abs_adjcorr  # noqa

BETA, GROUP = 0.5, 128
SPS = [0.4, 0.5, 0.6, 0.7, 0.8]


def group_kurt(W_norm, mask, gsize):
    K, N = W_norm.shape
    if not (N > gsize and N % gsize == 0):
        return None
    Wg = W_norm.view(K, N // gsize, gsize); Mg = mask.view(K, N // gsize, gsize).bool()
    cnt = Mg.sum(-1, keepdim=True).clamp(min=1.0)
    mu = torch.where(Mg, Wg, torch.zeros_like(Wg)).sum(-1, keepdim=True) / cnt
    Wc = torch.where(Mg, Wg - mu, torch.zeros_like(Wg))
    var = (Wc * Wc).sum(-1, keepdim=True) / cnt
    m4 = (Wc ** 4).sum(-1, keepdim=True) / cnt
    kurt = (m4 / var.clamp(min=1e-12) ** 2)
    valid = Mg.any(-1).squeeze(-1) if False else (cnt.squeeze(-1) > 3)
    k = kurt.squeeze(-1)[valid]
    return float(k.mean().item()) if k.numel() else None


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
    print(f"\n######## {MODEL} sample={sample} g={GROUP} beta={BETA} ########", flush=True)
    kacc = {sp: [0.0, 0] for sp in SPS}; cacc = {sp: [0.0, 0] for sp in SPS}
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
            X = A[key].to(device)
            for sp in SPS:
                W_comp, mask = ns.balanced_mask_and_obs(W, X, sp, device, col_exp=BETA)
                r, c = ns.compute_norm_scales(W_comp, mask, 'col', device)
                W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
                kv = group_kurt(W_norm, mask, block)
                cv = mean_abs_adjcorr(W_norm, mask, block)
                if kv is not None: kacc[sp][0] += kv; kacc[sp][1] += 1
                if cv is not None: cacc[sp][0] += cv; cacc[sp][1] += 1
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache()
    print(f"  survivor per-group KURTOSIS (uniform=1.8, gauss=3.0) and mean|adj corr| (indep~0) vs sparsity:", flush=True)
    print(f"    {'sp':<6}{'kurtosis':<12}{'|adj corr|':<12}", flush=True)
    for sp in SPS:
        k = kacc[sp][0]/max(1, kacc[sp][1]); cc = cacc[sp][0]/max(1, cacc[sp][1])
        print(f"    {sp:<6}{k:<12.3f}{cc:<12.4f}", flush=True)


def main():
    P = argparse.ArgumentParser(); P.add_argument("--models", default="gemma-2b,tinyllama,qwen-1.5b")
    Aa = P.parse_args()
    for m in Aa.models.split(","): run_model(m.strip())


if __name__ == "__main__":
    main()
