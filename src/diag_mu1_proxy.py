#!/usr/bin/env python3
"""Per-row inverse-μ (PPL 116) beats PRISM; its within-row lever is /μ1 (Sinkhorn col
scale). /μ1 HELPS but /col_std HURTS (cell G), so μ1 ≠ col_std. Find a NON-Sinkhorn
column scale that proxies μ1's *ordering* (what a per-row threshold uses), so we can
replicate the winning within-row reweighting without Sinkhorn. Cheap: masks only, no eval."""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink  # noqa
from sinq.sparse_quant import sinkhorn_log_sparse_aware  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"; SP = 0.7033492822966506


def logcorr(a, b):
    a = a.float().clamp(min=1e-12).log(); b = b.float().clamp(min=1e-12).log()
    a = a - a.mean(); b = b - b.mean()
    return (a * b).mean().item() / ((a.std() + 1e-9) * (b.std() + 1e-9))


def rankcorr(a, b):  # Spearman-ish: Pearson on ranks
    ra = a.float().argsort().argsort().float(); rb = b.float().argsort().argsort().float()
    ra = (ra - ra.mean()) / (ra.std() + 1e-9); rb = (rb - rb.mean()) / (rb.std() + 1e-9)
    return (ra * rb).mean().item()


def main():
    name = bs.MODELS["gemma-2b"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, DEV)
    acts = bs.collect_activations(model, cal, DEV)
    layers = bs.get_transformer_layers(model)
    paths = {"q": "self_attn.q_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj",
             "down": "mlp.down_proj", "k": "self_attn.k_proj"}
    print(f"{'mat(L)':10s} | rank-corr(μ1 , proxy):  {'col_std':>8s} {'col_L2':>8s} {'col_mad':>8s} {'dual_c':>8s} {'act||X||':>8s}")
    for li in [3, 9, 15]:
        layer = layers[li].to(DEV)
        for t, ap in paths.items():
            parts = ap.split('.'); mod = layer
            for p in parts: mod = getattr(mod, p)
            W = mod.weight.data.to(DEV).float()
            X = acts.get(f"layer_{li}.{ap}", None)
            if X is None: continue
            # inverse-μ mask + its final sparse-aware μ1 (as PRISM computes it)
            _, mask = nosink.inverse_mu_mask_and_obs(W.clone(), X.to(DEV), SP, DEV)
            _, mu1, _ = sinkhorn_log_sparse_aware(W * mask, mask, order=16)
            mu1 = mu1.view(-1).float().clamp(min=1e-12)                 # [N] column scale
            # non-Sinkhorn column-scale proxies (survivor-aware where natural)
            col_std = nosink._masked_std(W, mask, dim=0)                # [N]
            col_L2 = (W * mask).norm(dim=0).clamp(min=1e-12)            # [N]
            col_mad = ((W.abs() * mask).sum(0) / mask.sum(0).clamp(min=1)).clamp(min=1e-12)
            rstd = nosink._masked_std(W, mask, dim=1).clamp(min=1e-8)   # [K]
            dual_c = nosink._masked_std(W / rstd.view(-1, 1), mask, dim=0)  # row-adjusted col std
            Xf = X.to(DEV).float();  Xf = Xf.reshape(-1, Xf.shape[-1])[:256]
            actn = Xf.norm(dim=0).clamp(min=1e-12)                      # [N] ||X|| per col
            print(f"{t+'('+str(li)+')':10s} | {'':22s}"
                  f"{rankcorr(mu1,col_std):8.2f} {rankcorr(mu1,col_L2):8.2f} "
                  f"{rankcorr(mu1,col_mad):8.2f} {rankcorr(mu1,dual_c):8.2f} {rankcorr(mu1,actn):8.2f}")
        bs.set_transformer_layer(model, li, layer.cpu())


if __name__ == "__main__":
    main()
