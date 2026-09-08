#!/usr/bin/env python3
"""Is CoBALT survivor near-UNIFORMITY a NORMALIZATION artifact? (tests the conceded gap)

The foreclosure rests on 'survivors near-uniform => RTN near-optimal', measured under ONE
in-scope normalization (c = per-column weight-std). The normalization is itself part of the
encoding. This probe measures the per-group-affine-normalized survivor KURTOSIS under
several in-scope c choices; if some c makes survivors NON-uniform (kurtosis far from 1.8),
grid-shaping could help THERE and the foreclosure is normalization-specific. If uniform
under all, the foreclosure is robust to the normalization frame.
"""
import os, sys
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs, nosink as ns
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemma-2b"
NBITS, GROUP, SP, BETA = 3, 64, 0.5, 0.5


def group_kurt(W_norm, mask, gsize):
    K, N = W_norm.shape
    if not (N > gsize and N % gsize == 0):
        return None
    Wg = W_norm.view(K, N // gsize, gsize); Mg = mask.view(K, N // gsize, gsize).bool
    cnt = Mg.sum(-1, keepdim=True).clamp(min=1.0)
    mu = torch.where(Mg, Wg, torch.zeros_like(Wg)).sum(-1, keepdim=True) / cnt
    A = torch.where(Mg, (Wg - mu).abs, torch.zeros_like(Wg)).amax(-1, keepdim=True).clamp(min=1e-8)
    t = ((Wg - mu) / A).clamp(-1, 1)
    return t[Mg].detach


def main:
    device = "cuda"
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    acts_all = bs.collect_activations(model, cal, device)
    for _l in ns.get_layers(model): _l.to("cpu")
    torch.cuda.empty_cache
    layer_paths = bs.get_layer_paths(model)
    n_layers = len(ns.get_layers(model))
    sample = sorted(set([0, n_layers // 2, n_layers - 1]))
    NORMS = ["none", "col", "acol"]
    pools = {nm: [] for nm in NORMS}
    layers = ns.get_layers(model)
    for li in sample:
        layer = layers[li].to(device)
        for ap in layer_paths:
            parts = ap.split('.'); parent = layer; ok = True
            for p in parts[:-1]:
                if not hasattr(parent, p): ok = False; break
                parent = getattr(parent, p)
            if not ok or not hasattr(parent, parts[-1]): continue
            lin = getattr(parent, parts[-1])
            if not isinstance(lin, torch.nn.Linear): continue
            acts = acts_all.get(f'layer_{li}.{ap}')
            if acts is None: continue
            W = lin.weight.data.clone
            W_comp, mask = ns.balanced_mask_and_obs(W, acts.to(device), SP, device, col_exp=BETA)
            Xa = acts.to(device).float
            if Xa.dim == 3: Xa = Xa.reshape(-1, Xa.shape[-1])
            act_abs = Xa[:min(Xa.shape[0], 256)].abs.mean(0)
            for nm in NORMS:
                r, c = ns.compute_norm_scales(W_comp, mask, nm, device, act_abs=act_abs, awq_alpha=0.5)
                W_norm = W_comp / (r.view(-1, 1) * c.view(1, -1))
                t = group_kurt(W_norm, mask, GROUP)
                if t is not None: pools[nm].append(t)
        layers[li] = layer.to("cpu"); torch.cuda.empty_cache

    def kurt(x):
        x = x.float; m = x.mean; s = x.std.clamp(min=1e-12)
        return float(((x - m) / s).pow(4).mean)
    print(f"\n[{MODEL}] per-group-affine survivor kurtosis under in-scope normalizations "
          f"(uniform=1.8, gaussian=3.0):", flush=True)
    for nm in NORMS:
        T = torch.cat(pools[nm])
        print(f"  c={nm:<5} kurtosis={kurt(T):.3f}  std={T.std:.4f}  n={T.numel}", flush=True)
    print("[verdict] if all ~1.8 => uniformity ROBUST to normalization frame (foreclosure holds "
          "for grid-shaping); if some >>2.5 => a c exists where survivors are shapeable.", flush=True)


if __name__ == "__main__":
    main
