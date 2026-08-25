#!/usr/bin/env python3
"""Measure WHERE PRISM's 3-bit/70% damage lives, by module TYPE, at the
whole-model level. Metric = wikitext2 perplexity (n_test=20 x 2048), identical
across all runs.

Mirrors src/ppl_sparsity_sweep.py for model load (google/gemma-2b fp16),
calibration data, and test data (n_test=20). Reuses benchmark_suite's exact
quantization + perplexity code. Reloads a fresh fp16 model for EVERY run
(quantization is destructive).

A FILTERED PRISM apply (copied from benchmark_suite.apply_prism_quantization,
line 1897) adds a module-type filter: only modules whose type is in include_types
are quantized; all others are left as their original fp16 Linear (skipped
entirely -- their weights are not touched). Everything else (activation
collection ONCE, sparse_quantize_sinq call, nbits=3, sparsity=0.70) is IDENTICAL.

RUNS (16): fp16 (no quant), full (all 7 types), isolated_T (7), loo_T (7).
Types = {q,k,v,o,gate,up,down}. Writes results/prism_sensitivity/sens.csv with
columns: run_name, mode, type, ppl.
"""
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
SPARSITY = 0.70
N_TEST = 20
TYPES = ["q", "k", "v", "o", "gate", "up", "down"]


def type_of(attr_path: str):
    """Derive module type from attr_path suffix (e.g. 'self_attn.q_proj'->'q')."""
    suffix = attr_path.split('.')[-1]          # q_proj, gate_proj, ...
    return suffix[:-len('_proj')] if suffix.endswith('_proj') else suffix


def apply_prism_quantization_filtered(
    model,
    calibration_data,
    nbits: int,
    sparsity: float,
    include_types: set,
    device: str = 'cuda',
) -> nn.Module:
    """COPY of benchmark_suite.apply_prism_quantization (line 1897) with ONE
    addition: a module-type filter. Only modules whose type is in include_types
    are quantized; every other module is left as its original fp16 Linear
    (skipped entirely -- weights untouched). Activation collection (ONCE),
    sparse_quantize_sinq call, nbits, sparsity are IDENTICAL to the original.
    """
    # Detect architecture type
    is_prenorm = bs.is_prenorm_architecture(model)
    if is_prenorm:
        print(f"  [PRISM] Detected pre-norm architecture - using simplified importance weighting")

    # First collect activations
    layer_activations = bs.collect_activations(model, calibration_data, device)
    layer_paths = bs.get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(bs.get_transformer_layers(model), desc=f"PRISM {nbits}b/{int(sparsity*100)}%")):
        layer = layer.to(device)

        for attr_path in layer_paths:
            # --- ONLY ADDITION: module-type filter -----------------------
            if type_of(attr_path) not in include_types:
                continue  # leave this module as its original fp16 Linear
            # -------------------------------------------------------------
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

            W = linear.weight.data.clone()
            bias = linear.bias.data.clone() if linear.bias is not None else None

            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)
            if activations is not None:
                activations = activations.to(device)

            # Apply PRISM: iterative refinement + OBS + sparse-aware Sinkhorn
            # Architecture-aware: use simplified importance for pre-norm (OPT)
            W_q, scales, zeros, mask, scale2, meta = bs.sparse_quantize_sinq(
                W, activations,
                sparsity=sparsity,
                nbits=nbits,
                method='sinq_wanda_inverse',  # Inverse-μ importance (post-norm only)
                device=device,
                use_compensation=True if activations is not None and sparsity > 0 else False,
                compensation_mode='prism',  # PRISM's sparse-aware Sinkhorn
                is_prenorm=is_prenorm  # architecture-aware handling
            )

            new_layer = bs.SparseQuantLinear(W_q, scales, zeros, mask, scale2, bias, meta)
            new_layer = new_layer.to(device)
            setattr(parent, parts[-1], new_layer)

            del W, linear
            if activations is not None:
                del activations
            torch.cuda.empty_cache()

        bs.set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def build_runs():
    runs = []
    runs.append(("fp16", "fp16", "", None))
    runs.append(("full", "full", "", set(TYPES)))
    for t in TYPES:
        runs.append((f"isolated_{t}", "isolated", t, {t}))
    for t in TYPES:
        runs.append((f"loo_{t}", "loo", t, set(TYPES) - {t}))
    return runs


def main():
    device = "cuda"
    name = bs.MODELS[MODEL]

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    test = bs.get_test_data(tok, seq_len=bs.EVAL_CONFIG["seq_len"],
                            n_samples=N_TEST, dataset_key="wikitext2")
    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"],
                                  dataset_key="wikitext2")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "prism_sensitivity")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "sens.csv")

    rows = []
    for run_name, mode, type_, include_types in build_runs():
        try:
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float16, device_map="cpu",
                trust_remote_code=True, low_cpu_mem_usage=True)
            bs.move_embed_to_device(model, device)
            if mode == "fp16":
                # No quantization: the filtered apply is skipped, so move the
                # whole model onto the device (the quant path is what otherwise
                # moves each transformer layer to the GPU).
                model = model.to(device)
            else:
                model = apply_prism_quantization_filtered(
                    model, cal, NBITS, SPARSITY, include_types, device)
            bs.move_final_layers_to_device(model, device)
            model.eval()
            ppl = bs.evaluate_perplexity(model, test, device)
            print(f"RESULT run={run_name} mode={mode} type={type_} ppl={ppl:.4f}", flush=True)
            rows.append((run_name, mode, type_, f"{ppl:.4f}"))
            del model
        except Exception as e:  # noqa: BLE001
            print(f"FAIL run={run_name} err={type(e).__name__}: {e}", flush=True)
            rows.append((run_name, mode, type_, f"ERROR: {type(e).__name__}: {e}"))
        gc.collect()
        torch.cuda.empty_cache()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_name", "mode", "type", "ppl"])
        w.writerows(rows)
    print(f"WROTE {out_csv}", flush=True)


if __name__ == "__main__":
    main()
