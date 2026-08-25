#!/usr/bin/env python3
"""Structural probe of PRISM's 3-bit / 0.70-sparsity format on gemma-2b.

Loads fp16 google/gemma-2b via the SAME path benchmark_suite uses, applies PRISM
by calling the SAME function (apply_prism_quantization), then reads out per-matrix
structural facts from the resulting SparseQuantLinear buffers + captured meta dict.

Does NOT reimplement or alter any quantization math: it only inspects the objects
apply_prism_quantization produced.
"""
import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
import benchmark_suite as bs  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# Map the gemma attr_path -> short module name requested in the CSV.
MODULE_SHORT = {
    'self_attn.q_proj': 'q',
    'self_attn.k_proj': 'k',
    'self_attn.v_proj': 'v',
    'self_attn.o_proj': 'o',
    'mlp.gate_proj': 'gate',
    'mlp.up_proj': 'up',
    'mlp.down_proj': 'down',
}

NBITS = 3
SPARSITY = 0.70
MODEL_KEY = 'gemma-2b'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "prism_probe")
OUT_CSV = os.path.join(OUT_DIR, "meta.csv")


def main():
    dev = 'cuda'
    name = bs.MODELS[MODEL_KEY]

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cal = bs.get_calibration_data(tok, n_samples=bs.EVAL_CONFIG["n_calibration_samples"],
                                  seq_len=bs.EVAL_CONFIG["calibration_seq_len"],
                                  dataset_key="wikitext2")

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Same load path as ppl_sparsity_sweep.py / benchmark_suite.
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True)
    bs.move_embed_to_device(model, dev)

    # SAME function benchmark_suite uses. No quantization math is touched.
    model = bs.apply_prism_quantization(model, cal, NBITS, SPARSITY, dev)
    bs.move_final_layers_to_device(model, dev)
    model.eval()

    layer_paths = bs.get_layer_paths(model)
    layers = bs.get_transformer_layers(model)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for layer_idx, layer in enumerate(layers):
        for attr_path in layer_paths:
            parts = attr_path.split('.')
            parent = layer
            try:
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                mod = getattr(parent, parts[-1])
            except AttributeError:
                continue
            if not isinstance(mod, bs.SparseQuantLinear):
                continue

            meta = mod.meta
            K, N = meta['shape']

            # achieved_sparsity: fraction of EXACT zeros in the dequantized/masked weight,
            # using the SAME dequant path the forward pass uses.
            with torch.no_grad():
                W_deq = bs.dequantize_sparse_sinq(
                    mod.W_q, mod.scales, mod.zeros, mod.mask, mod.scale2, meta
                ).float()
                achieved_sparsity = (W_deq == 0).float().mean().item()

            rows.append({
                'layer_idx': layer_idx,
                'module': MODULE_SHORT.get(attr_path, attr_path),
                'K_out': int(K),
                'N_in': int(N),
                'group_size': int(meta['group_size']),
                'requested_nbits': int(meta['requested_nbits']),
                'actual_nbits': int(meta['nbits']),
                'achieved_sparsity': achieved_sparsity,
                'n_scale_elems': int(mod.scales.numel()),
                'n_zero_elems': int(mod.zeros.numel()),
                'n_scale2_elems': int(mod.scale2.numel()),
                'n_mask_elems': int(mod.mask.numel()),
            })

    fields = ['layer_idx', 'module', 'K_out', 'N_in', 'group_size',
              'requested_nbits', 'actual_nbits', 'achieved_sparsity',
              'n_scale_elems', 'n_zero_elems', 'n_scale2_elems', 'n_mask_elems']
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # -------- compact summary --------
    n_total = len(rows)

    diff = [r for r in rows if r['actual_nbits'] != r['requested_nbits']]

    sps = [r['achieved_sparsity'] for r in rows]
    sp_min = min(sps)
    sp_max = max(sps)
    sp_mean = sum(sps) / len(sps)

    shape_counts = {}
    for r in rows:
        key = (r['K_out'], r['N_in'], r['group_size'])
        shape_counts[key] = shape_counts.get(key, 0) + 1

    gsizes = sorted({r['group_size'] for r in rows})

    print("==== PRISM PROBE SUMMARY (gemma-2b, nbits=3, sparsity=0.70) ====")
    print(f"(a) total_target_matrices = {n_total}")
    print(f"(b) count_actual_ne_requested = {len(diff)}")
    for r in diff:
        print(f"     layer {r['layer_idx']} {r['module']}: actual_nbits={r['actual_nbits']} (requested={r['requested_nbits']})")
    print(f"(c) achieved_sparsity  min={sp_min:.6f}  mean={sp_mean:.6f}  max={sp_max:.6f}")
    print("(d) distinct (K_out, N_in, group_size) shapes with counts:")
    for key in sorted(shape_counts):
        print(f"     (K_out={key[0]}, N_in={key[1]}, group_size={key[2]}): {shape_counts[key]}")
    print(f"(e) group_size value(s) used = {gsizes}")
    print(f"CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
