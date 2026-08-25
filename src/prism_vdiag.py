#!/usr/bin/env python3
"""Focused diagnostic of WHY v_proj fails under PRISM 3-bit/70%.

ALL runs are v-only (include_types={"v"}); every other module is left as its
original fp16 Linear. Mirrors src/prism_sensitivity.py for model load
(google/gemma-2b fp16), calibration data, and test data (n_test=20), and reuses
its filtered PRISM apply. Metric = wikitext2 perplexity (n_test=20 x 2048),
identical across runs.

Two added capabilities:
  (1) nbits is variable per run (plumbed straight into the existing filtered
      apply, which already takes nbits and forwards it to bs.sparse_quantize_sinq).
  (2) A PRUNE-ONLY mode for v: for each v_proj Linear, compute the Wanda per-row
      mask from collected calibration activations and zero the pruned weights,
      leaving the module as fp16 nn.Linear (NO quantization, NO SparseQuantLinear).

Writes one row per run to results/prism_vdiag/vdiag.csv with columns:
  run_name, nbits, sparsity, ppl

--only <run_name> runs just that one run (smoke test); no arg runs all.
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

# Reuse the working filtered apply + type_of helper verbatim.
from prism_sensitivity import apply_prism_quantization_filtered, type_of  # noqa: E402

MODEL = "gemma-2b"
N_TEST = 20
INCLUDE_TYPES = {"v"}


def apply_prune_only_v(model, calibration_data, sparsity: float, device: str = 'cuda') -> nn.Module:
    """PRUNE-ONLY mode: for each v_proj Linear, compute the Wanda per-row mask
    from calibration activations and zero the pruned weights IN PLACE, leaving
    the module as a plain fp16 nn.Linear. Only v_proj is touched; all else fp16.
    """
    layer_activations = bs.collect_activations(model, calibration_data, device)
    layer_paths = bs.get_layer_paths(model)

    for layer_idx, layer in enumerate(tqdm(bs.get_transformer_layers(model), desc=f"PRUNE-ONLY v {int(sparsity*100)}%")):
        layer = layer.to(device)

        for attr_path in layer_paths:
            if type_of(attr_path) not in INCLUDE_TYPES:
                continue  # leave this module as its original fp16 Linear
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

            W = linear.weight.data
            act_key = f'layer_{layer_idx}.{attr_path}'
            activations = layer_activations.get(act_key, None)
            if activations is None:
                continue  # no calibration signal -> leave untouched

            scaler_row = bs._wanda_scaler_row(activations, device)
            mask = bs._wanda_row_mask(W, scaler_row, sparsity)
            linear.weight.data = (W * mask.to(W.device, W.dtype))

            del activations
            torch.cuda.empty_cache()

        bs.set_transformer_layer(model, layer_idx, layer)
        gc.collect()
        torch.cuda.empty_cache()

    return model


def build_runs():
    """(run_name, mode, nbits, sparsity). mode in {"prism", "pruneonly"}."""
    runs = []
    runs.append(("isolated_v_n3_s70", "prism", 3, 0.70))
    runs.append(("isolated_v_n4_s70", "prism", 4, 0.70))
    runs.append(("isolated_v_n5_s70", "prism", 5, 0.70))
    runs.append(("isolated_v_n6_s70", "prism", 6, 0.70))
    runs.append(("isolated_v_n8_s70", "prism", 8, 0.70))
    runs.append(("v_quantonly_n3_s0", "prism", 3, 0.0))
    runs.append(("v_pruneonly_s70", "pruneonly", None, 0.70))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None,
                        help="Run only the named run (smoke test).")
    args = parser.parse_args()

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
                           "results", "prism_vdiag")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "vdiag.csv")

    all_runs = build_runs()
    if args.only is not None:
        all_runs = [r for r in all_runs if r[0] == args.only]
        if not all_runs:
            print(f"ERROR: no run named '{args.only}'", flush=True)
            sys.exit(1)

    rows = []
    for run_name, mode, nbits, sparsity in all_runs:
        try:
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch.float16, device_map="cpu",
                trust_remote_code=True, low_cpu_mem_usage=True)
            bs.move_embed_to_device(model, device)

            if mode == "pruneonly":
                model = apply_prune_only_v(model, cal, sparsity, device)
            else:
                model = apply_prism_quantization_filtered(
                    model, cal, nbits, sparsity, INCLUDE_TYPES, device)

            bs.move_final_layers_to_device(model, device)
            model.eval()
            ppl = bs.evaluate_perplexity(model, test, device)
            nbits_str = "" if nbits is None else str(nbits)
            print(f"RESULT run={run_name} nbits={nbits_str} sparsity={sparsity} ppl={ppl:.4f}", flush=True)
            rows.append((run_name, nbits_str, sparsity, f"{ppl:.4f}"))
            del model
        except Exception as e:  # noqa: BLE001
            print(f"FAIL run={run_name} err={type(e).__name__}: {e}", flush=True)
            rows.append((run_name, "" if nbits is None else str(nbits), sparsity,
                         f"ERROR: {type(e).__name__}: {e}"))
        gc.collect()
        torch.cuda.empty_cache()

    # When smoke-testing a single run, don't clobber a full CSV.
    write_mode = "a" if (args.only is not None and os.path.exists(out_csv)) else "w"
    with open(out_csv, write_mode, newline="") as f:
        w = csv.writer(f)
        if write_mode == "w":
            w.writerow(["run_name", "nbits", "sparsity", "ppl"])
        w.writerows(rows)
    print(f"WROTE {out_csv}", flush=True)


if __name__ == "__main__":
    main()
