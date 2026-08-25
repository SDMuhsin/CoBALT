#!/usr/bin/env python3
"""GROUNDING measurement for the from-scratch joint operator. The activation Hessian
H = XᵀX (N×N) has rank ≤ S=256 (256 calib rows), so its nonzero spectrum = eigenvalues
of the cheap 256×256 Gram G = XXᵀ. Measure the decay: how many directions carry the
energy? This decides the rank r of a robust low-rank-Hessian operator (vs diagonal =
Wanda, which is blind to correlations, and full H⁻¹ = OBS, which is noise-dominated).
Also: how MUCH off-diagonal correlation is there (||H||_off / ||H||_diag)? If tiny,
diagonal (Wanda) is already near-optimal and correlations won't help — a key go/no-go."""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"


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
    paths = {"q": "self_attn.q_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "down": "mlp.down_proj"}
    print(f"{'mat(L)':10s} {'N':>6s} {'S':>5s} | {'rank90':>7s} {'rank99':>7s} | {'topEV%':>7s} | corr_ratio=||H_off||/||H_diag||")
    for li in [3, 9, 15]:
        layer = layers[li].to(DEV)
        for t, ap in paths.items():
            X = acts.get(f"layer_{li}.{ap}", None)
            if X is None: continue
            X = X.float().to(DEV)
            if X.dim() == 3: X = X.reshape(-1, X.shape[-1])
            X = X[:min(X.shape[0], 256)]                       # [S, N]
            S, N = X.shape
            G = X @ X.t()                                      # [S,S] Gram, same nonzero spectrum as XᵀX
            evals = torch.linalg.eigvalsh(G).flip(0).clamp(min=0)   # descending
            tot = evals.sum().clamp(min=1e-9)
            csum = torch.cumsum(evals, 0) / tot
            r90 = int((csum < 0.90).sum()) + 1
            r99 = int((csum < 0.99).sum()) + 1
            top = (evals[0] / tot).item() * 100
            # off-diagonal magnitude of the true Hessian H = XᵀX (diag vs full), on a column subsample
            idx = torch.randperm(N, device=DEV)[:min(N, 2048)]
            Xs = X[:, idx]
            H = Xs.t() @ Xs                                    # [n,n]
            d = H.diag(); off = H - torch.diag(d)
            corr_ratio = (off.norm() / (torch.diag(d).norm() + 1e-9)).item()
            print(f"{t+'('+str(li)+')':10s} {N:6d} {S:5d} | {r90:7d} {r99:7d} | {top:6.1f}% | {corr_ratio:.3f}")
        bs.set_transformer_layer(model, li, layer.cpu())


if __name__ == "__main__":
    main()
