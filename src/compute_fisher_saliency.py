#!/usr/bin/env python3
"""Precompute the EXACT diagonal empirical-Fisher activation statistic M_ij = E[g_i² x_j²] for
every target linear (g = ∂L/∂y output grad, x = layer input), cached per matrix for the
fisher_sal mask in nosink.py.

Why this is the deep use of the output Fisher G (see llmdocs/CONTEXT.md):
  The per-weight empirical Fisher diagonal is F_ij = E[(∂L/∂w_ij)²] = E[(g_i x_j)²] = E[g_i² x_j²].
  The 2nd-order pruning saliency of w_ij is ½ F_ij w_ij². Two known approximations DROP structure:
    • Wanda:      F_ij ≈ E[x_j²]           (drops g entirely ⇒ assumes output Fisher G = I)
    • factorized: F_ij ≈ E[g_i²]·E[x_j²]   (K-FAC independence ⇒ s_i only reweights ALLOCATION)
  The EXACT joint E[g_i² x_j²] keeps the (g_i², x_j²) correlation ⇒ it reweights COLUMNS
  DIFFERENTLY PER ROW — the within-row selection lever Wanda cannot have (and where the measured
  downstream headroom lives: the F→C gap was within-row, not allocation). Fully non-Sinkhorn.

Budget: ONE forward+backward pass over 8×512 calib tokens (a gradient, not an optimization step;
≤ PRISM's 8 forward passes). Accumulate M += (g²)ᵀ(x²) per layer (a matmul) into CPU fp32, save
fp16 per matrix. Model in float32 for accurate grads (measurement only)."""
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
OUT_DIR = os.path.join(_ROOT, "results", "fisher_sal")


def main():
    name = bs.MODELS["gemma-2b"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=N_CALIB, seq_len=SEQ_LEN, dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32,
                                                 device_map=DEV, low_cpu_mem_usage=True)
    model.eval()

    layers = bs.get_transformer_layers(model)
    paths = bs.get_layer_paths(model)
    x_store = {}       # key -> current-batch input x²  [T, N] (from forward hook)
    M = {}             # key -> CPU fp32 accumulator [K, N]
    cnt = {}           # key -> token count
    fh, bh = [], []

    def mk_fwd(key):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()   # [T, N]
            x_store[key] = x * x                                        # x², keep for backward
        return hook

    def mk_bwd(key):
        def hook(mod, gin, gout):
            g = gout[0]
            if g is None or key not in x_store:
                return
            g2 = g.detach().reshape(-1, g.shape[-1]).float()           # [T, K]
            g2 = g2 * g2
            x2 = x_store.pop(key)                                       # [T, N]
            m = (g2.t() @ x2)                                           # [K, N] = Σ_t g_i² x_j²
            prev = M.get(key)
            M[key] = m.cpu() if prev is None else prev + m.cpu()
            cnt[key] = cnt.get(key, 0) + g2.shape[0]
        return hook

    key_of = {}
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
                key = f"layer_{li}.{ap}"
                key_of[mod] = key
                fh.append(mod.register_forward_hook(mk_fwd(key)))
                bh.append(mod.register_full_backward_hook(mk_bwd(key)))

    for seq in cal:
        ids = seq.to(DEV) if torch.is_tensor(seq) else torch.tensor(seq, device=DEV)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)
        out.loss.backward()
        x_store.clear()
    for h in fh + bh:
        h.remove()

    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for key, m in M.items():
        mi = (m / max(1, cnt[key])).clamp(min=0).float()              # E[g_i² x_j²]
        # KEEP fp32: values are ~1e-5 (fp16 min-normal ~6e-5 ⇒ mass underflows to 0 ⇒ collapse).
        # Mask cares only about RELATIVE saliency, so normalize per matrix to O(1) for clean fp32.
        mi = mi / (mi.mean() + 1e-30)
        torch.save(mi, os.path.join(OUT_DIR, key + ".pt"))
        n += 1
    print(f"SAVED {OUT_DIR}  n_matrices={n}", flush=True)


if __name__ == "__main__":
    main()
