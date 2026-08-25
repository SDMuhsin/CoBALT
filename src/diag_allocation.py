#!/usr/bin/env python3
"""Understand PRISM's inverse-μ mask ALLOCATION so a non-Sinkhorn mask can mimic it.
Per-row Wanda matches PRISM on PPL but loses downstream — the difference is that
inverse-μ allocates survivors NON-uniformly across output rows while per-row is
uniform. Here we measure, on real gemma-2b matrices at the v-dense plan sparsity:
  - the per-row keep-fraction distribution under inverse-μ (how non-uniform? starved?)
  - what row statistic predicts it (row_std / row L2 / row max) → the budget signal.
Cheap: activations (1 pass) + per-matrix Sinkhorn mask; NO eval."""
import os, sys
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "src"))
import benchmark_suite as bs  # noqa
import nosink  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa

DEV = "cuda"; SP = 0.7033492822966506


def corr(a, b):
    a = a.float(); b = b.float()
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return (a * b).mean().item()


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
             "up": "mlp.up_proj", "down": "mlp.down_proj", "k": "self_attn.k_proj"}
    print(f"{'mat(layer)':16s} {'keepfrac: min':>13s} {'p10':>6s} {'med':>6s} {'max':>6s} {'std':>6s} {'%rows<10%':>9s} | corr(keep, row_std/L2/max)")
    for li in [3, 9, 15]:
        layer = layers[li].to(DEV)
        for t, ap in paths.items():
            parts = ap.split('.'); mod = layer
            for p in parts: mod = getattr(mod, p)
            W = mod.weight.data.to(DEV)
            ak = f"layer_{li}.{ap}"
            X = acts.get(ak, None)
            if X is None: continue
            _, mask = nosink.inverse_mu_mask_and_obs(W.clone().float(), X.to(DEV), SP, DEV)
            keep = mask.float().mean(dim=1)                    # [K] per-row keep fraction
            Wf = W.float()
            row_std = Wf.std(dim=1); row_l2 = Wf.norm(dim=1); row_max = Wf.abs().max(dim=1).values
            frac_starved = (keep < 0.10).float().mean().item()
            print(f"{t+'('+str(li)+')':16s} {keep.min():13.3f} {keep.quantile(0.1):6.3f} "
                  f"{keep.median():6.3f} {keep.max():6.3f} {keep.std():6.3f} {frac_starved*100:8.1f}% | "
                  f"{corr(keep,row_std):+.2f}/{corr(keep,row_l2):+.2f}/{corr(keep,row_max):+.2f}")
        bs.set_transformer_layer(model, li, layer.cpu())


if __name__ == "__main__":
    main()
