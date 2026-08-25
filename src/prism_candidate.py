#!/usr/bin/env python3
"""Candidate-method harness: PRISM quantization with PER-TYPE sparsity allocation.

Insight (from M-vdiag): v_proj's 3-bit/70% damage is 100% from PRUNING, not
quantization (3-bit DENSE v ≈ fp16; 70%-pruned v collapses). v_proj is the GQA
value bottleneck (256 out, 1 KV head shared by 8 q-heads). Fix: keep v_proj DENSE
(sparsity 0), still 3-bit quantized, compensate elsewhere to hold global 70%.

This harness uses the PRISM quantizer (Sinkhorn) ONLY to validate the sparsity-
reallocation INSIGHT end-to-end. The final deliverable must swap in a NON-Sinkhorn
base (req #3) — that is a separate step.

Modes:
  uniform : all types sparsity=0.70 (reproduces PRISM baseline under this script)
  vdense  : v_proj sparsity=0.0 (dense), all others 0.70
  custom  : pass --v-sparsity and --other-sparsity

Reports PPL (wikitext2, n_test) AND the analytic global sparsity (honest).
"""
import argparse
import csv
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
import benchmark_suite as bs  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from tqdm import tqdm  # noqa: E402
import torch.nn as nn  # noqa: E402

MODEL = "gemma-2b"
NBITS = 3
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]


def type_of(attr_path: str):
    suffix = attr_path.split('.')[-1]
    return suffix[:-len('_proj')] if suffix.endswith('_proj') else suffix


def apply_prism_per_type_sparsity(model, calibration_data, nbits, sparsity_by_type, device='cuda'):
    """PRISM quantize EVERY target linear, but with per-module-type sparsity.
    Verbatim PRISM loop (benchmark_suite.apply_prism_quantization) except sparsity
    is looked up per type. Returns (model, param_stats) where param_stats is a list
    of (type, numel, requested_sparsity) for honest global-sparsity accounting.
    """
    is_prenorm = bs.is_prenorm_architecture(model)
    layer_activations = bs.collect_activations(model, calibration_data, device)
    layer_paths = bs.get_layer_paths(model)
    stats = []
    for layer_idx, layer in enumerate(tqdm(bs.get_transformer_layers(model), desc=f"PRISM/pt {nbits}b")):
        layer = layer.to(device)
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                linear = getattr(parent, parts[-1])
            except AttributeError:
                continue
            if not isinstance(linear, nn.Linear):
                continue
            t = type_of(attr_path)
            sp = sparsity_by_type.get(t, 0.70)
            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None
            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)
            if activations is not None:
                activations = activations.to(device)
            W_q, scales, zeros, mask, scale2, meta = bs.sparse_quantize_sinq(
                W, activations,
                sparsity=sp, nbits=nbits,
                method='sinq_wanda_inverse', device=device,
                use_compensation=True if activations is not None and sp > 0 else False,
                compensation_mode='prism', is_prenorm=is_prenorm)
            new_layer = bs.SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)
            stats.append((t, W.numel(), sp))
            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()
        bs.set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()
    return model, stats


def global_sparsity(stats):
    tot = sum(n for _, n, _ in stats)
    pruned = sum(n * sp for _, n, sp in stats)
    return pruned / tot if tot else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="vdense", choices=["uniform", "vdense", "custom"])
    ap.add_argument("--v-sparsity", type=float, default=0.0)
    ap.add_argument("--other-sparsity", type=float, default=0.70)
    ap.add_argument("--ntest", type=int, default=20)
    args = ap.parse_args()

    if args.mode == "uniform":
        sp_by_type = {t: 0.70 for t in TYPES}
    elif args.mode == "vdense":
        sp_by_type = {t: 0.70 for t in TYPES}; sp_by_type["v"] = 0.0
    else:
        sp_by_type = {t: args.other_sparsity for t in TYPES}; sp_by_type["v"] = args.v_sparsity

    device = "cuda"
    name = bs.MODELS[MODEL]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["seq_len"], n_samples=args.ntest, dataset_key="wikitext2")
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"], dataset_key="wikitext2")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, device)
    model, stats = apply_prism_per_type_sparsity(model, cal, NBITS, sp_by_type, device)
    bs.move_final_layers_to_device(model, device)
    model.eval()
    ppl = bs.evaluate_perplexity(model, test, device)
    gsp = global_sparsity(stats)
    print(f"RESULT mode={args.mode} sp_by_type={sp_by_type} global_sparsity={gsp:.4f} "
          f"ntest={args.ntest} nbits={NBITS} ppl={ppl:.4f}", flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "prism_candidate")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "candidate.csv"), "a", newline="") as f:
        csv.writer(f).writerow([args.mode, f"{gsp:.4f}", args.ntest, NBITS, f"{ppl:.4f}", str(sp_by_type)])
    print(f"APPENDED {out_dir}/candidate.csv", flush=True)


if __name__ == "__main__":
    main()
