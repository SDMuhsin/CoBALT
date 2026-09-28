#!/usr/bin/env python3
"""MEASURE-FIRST for a NON-starvation ORTHOGONAL mask lever (starvation-rescue is saturated). Question:
does CoBALT's GLOBAL column-balanced mask leave HEAD/BLOCK structure it ignores? For o_proj the INPUT
channels are partitioned into attention heads (N = n_heads * head_dim). If, under the global balanced mask,
per-HEAD keep-rates are IMBALANCED (some heads' input block kept much more than others), a per-head
structured balance is an untried orthogonal lever. Reports, on o_proj matrices, the CV of per-head keep-
rate under the global balanced mask (high CV => head imbalance => room). Also q/k/v (output heads) via rows.
Cheap (masks only)."""
import os, sys, argparse
import torch
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "benchmarks")); sys.path.insert(0, _ROOT)
import benchmark_suite as bs  # noqa
import nosink as ns  # noqa
import torch.nn as nn  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa
DEV = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sparsity", type=float, default=0.6)
    args = ap.parse_args()
    name = bs.MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).config
    nheads = getattr(cfg, 'num_attention_heads', None)
    hid = getattr(cfg, 'hidden_size', None)
    hd = hid // nheads if (nheads and hid) else None
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV)
    acts = bs.collect_activations(model, cal, DEV)
    layers = ns.get_layers(model); paths = bs.get_layer_paths(model)
    print(f"# model={args.model} nheads={nheads} head_dim={hd} sp={args.sparsity}")
    sp = args.sparsity
    ohcv = []; ihcv = []
    for li in range(len(layers)):
        for ap_ in paths:
            X = acts.get(f'layer_{li}.{ap_}')
            if X is None:
                continue
            mod = layers[li]; ok = True
            for p in ap_.split('.'):
                if not hasattr(mod, p):
                    ok = False; break
                mod = getattr(mod, p)
            if not (ok and isinstance(mod, nn.Linear)):
                continue
            is_o = ap_.endswith('o_proj') or ap_.endswith('out_proj') or ap_.endswith('.o')
            W = mod.weight.data.clone().float().to(DEV)
            Xd = X.to(DEV)
            if Xd.dim() == 3:
                Xd = Xd.reshape(-1, Xd.shape[-1])
            Xd = Xd[:min(Xd.shape[0], 256)]
            _, mask = ns.balanced_mask_and_obs(W, Xd, sp, DEV, col_exp=0.5, no_obs=True)
            K, N = mask.shape
            if is_o and hd and N % hd == 0 and N // hd == nheads:
                # INPUT channels head-partitioned: per-head keep-rate = mean over that head's cols
                kr = mask.mean(0).view(nheads, hd).mean(1)     # [nheads]
                ihcv.append((kr.std() / (kr.mean() + 1e-9)).item())
            if hd and K % hd == 0 and (K // hd) == nheads and not is_o:
                # OUTPUT channels head-partitioned (q/k/v): per-head keep-rate = mean over that head's rows
                kro = mask.mean(1).view(nheads, hd).mean(1)
                ohcv.append((kro.std() / (kro.mean() + 1e-9)).item())
        torch.cuda.empty_cache()
    import statistics as st
    if ihcv:
        print(f"o_proj INPUT-head keep-rate CV: mean={st.mean(ihcv):.4f} max={max(ihcv):.4f} n={len(ihcv)}"
              f"  (high => global balance leaves head imbalance => per-head lever has room)")
    if ohcv:
        print(f"q/k/v OUTPUT-head keep-rate CV: mean={st.mean(ohcv):.4f} max={max(ohcv):.4f} n={len(ohcv)}")
    print("REF: a uniform-across-heads mask would have CV~0; CV>~0.1 = real head imbalance.")


if __name__ == "__main__":
    main()
