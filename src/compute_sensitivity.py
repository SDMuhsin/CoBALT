#!/usr/bin/env python3
"""Compute the OUTPUT-side Fisher diagonal (per-output-channel sensitivity) for EVERY
target linear, and cache it for the G-aware mask in nosink.py.

Derivation (see llmdocs/CONTEXT.md): the task-loss Hessian of a linear layer factorizes
(K-FAC) as H_act ⊗ G, G = E[g gᵀ] the output-gradient covariance (g = ∂L/∂y). Per-matrix
reconstruction assumes G = I. diag(G)_i = s_i = E[(∂L/∂y_i)²] is the per-output-channel
SENSITIVITY (how much output channel i matters to the LM loss). diag_output_fisher.py
measured s_i is FAR from I on every matrix (CV 0.3–7.5, max/med up to 1884). This script
caches s_i for all 126 matrices so the mask can weight per-row.

Budget: ONE backward pass over 8×512 calib tokens (a gradient, not an optimization step) —
honestly ≤ PRISM's 8 forward passes. Model in float32 for accurate grads (measurement only;
the deliverable quantizer runs fp16 as before)."""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"
N_CALIB = 8
SEQ_LEN = 512
OUT = os.path.join(_ROOT, "results", "sensitivity", "sens.pt")


def main():
    name = bs.MODELS["gemma-2b"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=N_CALIB, seq_len=SEQ_LEN, dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32,
                                                 device_map=DEV, low_cpu_mem_usage=True)
    model.eval()  # eval mode, grads still flow

    layers = bs.get_transformer_layers(model)
    paths = bs.get_layer_paths(model)
    sum_g2, cnt, hooks = {}, {}, []

    def mk_hook(key):
        def hook(mod, gin, gout):
            g = gout[0]
            if g is None:
                return
            g = g.detach().reshape(-1, g.shape[-1]).float()   # [tokens, K]
            sum_g2[key] = sum_g2.get(key, 0) + (g * g).sum(0)
            cnt[key] = cnt.get(key, 0) + g.shape[0]
        return hook

    for li, layer in enumerate(layers):
        for ap in paths:
            mod = layer
            ok = True
            for p in ap.split('.'):
                if not hasattr(mod, p):
                    ok = False
                    break
                mod = getattr(mod, p)
            if ok and isinstance(mod, nn.Linear):
                hooks.append(mod.register_full_backward_hook(mk_hook(f"layer_{li}.{ap}")))

    for seq in cal:
        ids = seq.to(DEV) if torch.is_tensor(seq) else torch.tensor(seq, device=DEV)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)
        out.loss.backward()
    for h in hooks:
        h.remove()

    sens = {k: (sum_g2[k] / max(1, cnt[k])).clamp(min=0).cpu() for k in sum_g2}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(sens, OUT)

    # sanity: CV distribution across matrices
    cvs = []
    for k, s in sens.items():
        cv = (s.std() / (s.mean() + 1e-12)).item()
        cvs.append(cv)
    cvs = torch.tensor(cvs)
    print(f"SAVED {OUT}  n_matrices={len(sens)}  CV: min={cvs.min():.2f} "
          f"med={cvs.median():.2f} max={cvs.max():.2f}", flush=True)


if __name__ == "__main__":
    main()
