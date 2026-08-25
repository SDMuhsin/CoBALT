#!/usr/bin/env python3
"""FROM-SCRATCH crux measurement: within a per-row budget (each output row keeps 30%),
does a CORRELATION-AWARE mask beat Wanda's DIAGONAL one on HELD-OUT output reconstruction?
Objective = ‖X(w-ŵ)‖² (exact model output error). corr_ratio>1 (measured) says Wanda leaves
structure on the table; but S=256<<N makes full H⁻¹ noisy. Test regularized OBS saliency
w²/[H_λ⁻¹]_jj at several damping λ vs Wanda |w|·‖X‖. Compensation = damped-OBS matched to
each mask (isolates the MASK). Fit/test split = generalization, not calib overfit.
Per-row budget removes the cross-row starvation confound → per-row reconstruction IS
model-aligned here."""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
from sinq.sinkhorn import sinkhorn_log  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"; SP = 0.70


def obs_inv(Xfit, lam_frac):
    # H = XᵀX (N×N) with damping λ = lam_frac * mean(diag H). Returns H_inv, diag.
    N = Xfit.shape[1]
    H = Xfit.t() @ Xfit
    d = H.diag().clamp(min=1e-8)
    lam = lam_frac * d.mean()
    H = H + lam * torch.eye(N, device=Xfit.device)
    Hinv = torch.linalg.inv(H)
    return Hinv, Hinv.diag().clamp(min=1e-12)


def per_row_mask(imp, keepfrac):
    K, N = imp.shape
    kp = int(N * (1 - keepfrac)) if False else int(N * SP)  # n prune per row
    thr = torch.kthvalue(imp, kp, dim=1, keepdim=True).values
    return (imp > thr).float()


def compensate(W, mask, Hinv, Hinv_d):
    # damped-OBS compensation for the given mask (per row), vectorized-ish.
    Wc = W.clone()
    for i in range(W.shape[0]):
        pruned = W[i] * (1.0 - mask[i])
        Wc[i] = W[i] * mask[i] + (-Hinv @ (pruned / Hinv_d)) * mask[i]
    return Wc


def held_out_err(W, Wc, Xtest):
    # mean over rows of ‖Xtest (w-ŵ)‖² / ‖Xtest w‖²  (relative output error)
    dW = (W - Wc)
    num = (Xtest @ dW.t()).pow(2).sum(0)          # [K]
    den = (Xtest @ W.t()).pow(2).sum(0).clamp(min=1e-9)
    return (num / den).mean().item()


def main():
    name = bs.MODELS["gemma-2b"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=32, seq_len=512, dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, DEV)
    acts = bs.collect_activations(model, cal, DEV)
    layers = bs.get_transformer_layers(model)
    paths = {"q": "self_attn.q_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "down": "mlp.down_proj"}
    print("held-out relative output err (lower=better);  cols = Wanda | OBSλ=0.01 | OBSλ=0.1 | OBSλ=1.0 | magnitude | inverse-mu")
    for li in [3, 9, 15]:
        layer = layers[li].to(DEV)
        for t, ap in paths.items():
            parts = ap.split('.'); mod = layer
            for p in parts: mod = getattr(mod, p)
            W = mod.weight.data.to(DEV).float()
            X = acts.get(f"layer_{li}.{ap}", None)
            if X is None: continue
            X = X.float().to(DEV)
            if X.dim() == 3: X = X.reshape(-1, X.shape[-1])
            n = X.shape[0]
            if n < 128: continue
            h = n // 2
            Xfit, Xtest = X[:h], X[h:2 * h]                      # fit/test split
            actn = Xfit.norm(dim=0)
            results = []
            # compensation uses a fixed moderate damping so only the MASK differs
            Hc, Hc_d = obs_inv(Xfit, 0.1)
            # Wanda
            mw = per_row_mask(W.abs() * actn.view(1, -1), 1 - SP)
            results.append(held_out_err(W, compensate(W, mw, Hc, Hc_d), Xtest))
            # OBS saliency at several damping
            for lf in [0.01, 0.1, 1.0]:
                _, hd = obs_inv(Xfit, lf)
                sal = (W ** 2) / hd.view(1, -1)
                m = per_row_mask(sal, 1 - SP)
                results.append(held_out_err(W, compensate(W, m, Hc, Hc_d), Xtest))
            # magnitude only
            mm = per_row_mask(W.abs(), 1 - SP)
            results.append(held_out_err(W, compensate(W, mm, Hc, Hc_d), Xtest))
            # inverse-mu (PRISM's mask): importance = |W|*||X|| / (mu1*mu2), mu from sinkhorn(W).
            # Under per-row budget only mu1 (col) matters within a row.
            _, mu1, mu2 = sinkhorn_log(W, order=16)
            imp_mu = W.abs() * actn.view(1, -1) / (mu1.view(1, -1) * mu2.view(-1, 1) + 1e-6)
            mmu = per_row_mask(imp_mu, 1 - SP)
            results.append(held_out_err(W, compensate(W, mmu, Hc, Hc_d), Xtest))
            print(f"{t+'('+str(li)+')':10s} " + " | ".join(f"{r:.4f}" for r in results))
        bs.set_transformer_layer(model, li, layer.cpu())


if __name__ == "__main__":
    main()
