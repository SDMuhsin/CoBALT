#!/usr/bin/env python3
"""DEEP from-scratch grounding: the task-loss Fisher for a linear layer is H_act ⊗ G,
G = E[g gᵀ] the OUTPUT-gradient covariance (g = ∂L/∂y). Per-matrix reconstruction assumes
G = I (all output channels equally important, uncorrelated) — the rule-#2 error. Measure
how far G is from I:
  - sensitivity sᵢ = diag(G) = E[(∂L/∂yᵢ)²]: coefficient of variation CV (how non-uniform),
    and max/median ratio. High CV ⇒ channel-sensitivity weighting is a real, missing lever.
  - off-diag(G) magnitude ‖G_off‖/‖G_diag‖ (for K≤2048 outputs) ⇒ gradient correlations.
One LM-loss backward pass on a few calib sequences (gradient, not optimization; ≤PRISM budget)."""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"
TARGETS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
           "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


def main():
    name = bs.MODELS["gemma-2b"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=8, seq_len=512, dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32, device_map=DEV, low_cpu_mem_usage=True)
    model.eval()  # eval mode but grads still flow
    layers = bs.get_transformer_layers(model)

    # accumulate per-channel sum(g^2) and (for small K) G = sum(g gᵀ), per (layer,type)
    acc_diag = {}   # key -> [K] running sum of g^2
    acc_cnt = {}    # key -> token count
    acc_G = {}      # key -> [K,K] running sum of g gᵀ (only K<=2048)
    hooks = []

    def mk_hook(key):
        def hook(mod, gin, gout):
            g = gout[0]
            if g is None: return
            g = g.detach().reshape(-1, g.shape[-1]).float()   # [tokens, K]
            d = (g * g).sum(0)
            acc_diag[key] = acc_diag.get(key, 0) + d
            acc_cnt[key] = acc_cnt.get(key, 0) + g.shape[0]
            if g.shape[1] <= 2048:
                acc_G[key] = acc_G.get(key, 0) + g.t() @ g
        return hook

    for li in [3, 9, 15]:
        layer = layers[li]
        for t, ap in TARGETS.items():
            parts = ap.split('.'); mod = layer
            for p in parts: mod = getattr(mod, p)
            hooks.append(mod.register_full_backward_hook(mk_hook(f"{t}({li})")))

    # forward+backward LM loss, one sequence at a time (bounded memory)
    for seq in cal:
        ids = seq.to(DEV) if torch.is_tensor(seq) else torch.tensor(seq, device=DEV)
        if ids.dim() == 1: ids = ids.unsqueeze(0)
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=ids)
        out.loss.backward()
    for h in hooks: h.remove()

    print(f"{'mat(L)':10s} {'K':>6s} | {'s_i CV':>7s} {'max/med':>8s} {'top1%_share':>11s} | {'||G_off||/||G_diag||':>18s}")
    for key in sorted(acc_diag.keys()):
        s = (acc_diag[key] / max(1, acc_cnt[key])).clamp(min=0)   # [K] sensitivity
        cv = (s.std() / (s.mean() + 1e-12)).item()
        mm = (s.max() / (s.median() + 1e-12)).item()
        srt = s.sort(descending=True).values
        top1 = srt[:max(1, len(srt)//100)].sum() / (s.sum() + 1e-12)
        off = "-"
        if key in acc_G:
            G = acc_G[key] / max(1, acc_cnt[key])
            d = G.diag(); offm = (G - torch.diag(d)).norm() / (torch.diag(d).norm() + 1e-12)
            off = f"{offm.item():.3f}"
        print(f"{key:10s} {len(s):6d} | {cv:7.2f} {mm:8.1f} {top1.item()*100:10.1f}% | {off:>18s}")


if __name__ == "__main__":
    main()
