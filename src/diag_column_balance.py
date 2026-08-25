#!/usr/bin/env python3
"""Mechanistic test: is inverse-μ's DOWNSTREAM-winning mask more COLUMN-BALANCED than the
saliency masks (Wanda, exact-Fisher)? Every saliency criterion (reconstruction/Wanda,
sensitivity, exact-Fisher) FAILS downstream despite matching/beating PPL — hypothesis: the
downstream lever is not saliency but BALANCE (inverse-μ's /μ1 down-weights high-norm columns
⇒ keeps a column-balanced survivor set; saliency masks concentrate survivors in high-|W|·‖X‖
columns ⇒ column starvation ⇒ worse generalization).

Measures, per matrix, the per-COLUMN keep-fraction distribution (CV, %near-dead cols) for:
  wanda(per_row) | inverse_mu(global, the downstream winner) | fisher_sal(per_row,λ=0.25).
Low column-CV / few dead cols = balanced. Rank ONLY confirms the mechanism; the verdict is the
end-to-end model (rule #2). Uses one activation-collection pass (like nosink)."""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
import nosink as ns  # noqa

DEV = "cuda"
SP = 0.70
FISHER_DIR = os.path.join(_ROOT, "results", "fisher_sal")
# representative matrices across types/depths
SEL = [(3, "self_attn.q_proj"), (9, "self_attn.o_proj"), (9, "mlp.gate_proj"),
       (9, "mlp.up_proj"), (15, "mlp.down_proj"), (3, "self_attn.v_proj")]


def colstats(mask):
    kf = mask.mean(0)                      # per-column keep fraction [N]
    cv = (kf.std() / (kf.mean() + 1e-12)).item()
    dead = (kf < 0.05).float().mean().item() * 100
    return cv, dead, kf.mean().item()


def rowstats(mask):
    kf = mask.mean(1)
    return (kf.std() / (kf.mean() + 1e-12)).item(), (kf < 0.05).float().mean().item() * 100


def main():
    name = bs.MODELS["gemma-2b"]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, DEV)
    acts = bs.collect_activations(model, cal, DEV)
    layers = ns.get_layers(model)

    print(f"{'matrix':22s} {'mask':16s} | {'col_CV':>6s} {'col_dead%':>8s} | {'row_CV':>6s} {'row_dead%':>8s}")
    for li, ap in SEL:
        layer = layers[li].to(DEV)
        mod = layer
        for p in ap.split('.'):
            mod = getattr(mod, p)
        W = mod.weight.data.clone()
        X = acts.get(f'layer_{li}.{ap}')
        if X is None:
            continue
        Xd = X.to(DEV)
        # three masks (mask only; ignore returned W_comp)
        _, m_wanda = ns.wanda_mask_and_obs(W, Xd, SP, DEV, scope='per_row')
        _, m_invmu = ns.inverse_mu_mask_and_obs(W, Xd, SP, DEV, scope='global')
        fm_path = os.path.join(FISHER_DIR, f'layer_{li}.{ap}.pt')
        fM = torch.load(fm_path, map_location='cpu') if os.path.exists(fm_path) else None
        _, m_fish = ns.fisher_sal_mask_and_obs(W, Xd, SP, DEV, scope='per_row', fisher_M=fM, shrink=0.25)
        tag = ap.split('.')[-1].replace('_proj', '') + f"({li})"
        for nm, m in [("wanda_perrow", m_wanda), ("invmu_global", m_invmu), ("fisher_l0.25", m_fish)]:
            ccv, cdead, _ = colstats(m)
            rcv, rdead = rowstats(m)
            print(f"{tag:22s} {nm:16s} | {ccv:6.2f} {cdead:7.1f}% | {rcv:6.2f} {rdead:7.1f}%", flush=True)
        del layer
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
