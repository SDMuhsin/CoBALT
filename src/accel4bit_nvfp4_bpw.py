#!/usr/bin/env python
"""Effective bits-per-weight of a compressed-tensors NVFP4 artifact (accel4bit ARM C).

Formula (quantized linears only):
    bpw_q = 8 * sum_bytes(weight_packed + weight_scale + weight_global_scale [+ input_global_scale])
            / sum(n_weights),   n_weights = rows * (packed_cols * 2)
i.e. 4 bits of FP4 code per weight + 8 bits FP8 per block of 16 (=0.5) + 32-bit global scale(s)
per tensor -> ~4.5 bpw.  Also reports the whole-checkpoint average over ALL parameters
(including un-quantized embeddings / norms / vision tower) for reference.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from safetensors import safe_open

DT_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "U8": 1, "I8": 1,
            "I32": 4, "I64": 8, "BOOL": 1, "F64": 8, "I16": 2, "U16": 2, "U32": 4}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--json", default=None)
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(a.dir, "*.safetensors")))
    per_layer = defaultdict(dict)
    total_bytes = 0
    total_params_all = 0
    for f in files:
        with safe_open(f, "pt") as sf:
            for k in sf.keys():
                sl = sf.get_slice(k)
                shape = sl.get_shape()
                dt = sl.get_dtype()
                n = 1
                for s in shape:
                    n *= s
                b = n * DT_BYTES[dt]
                total_bytes += b
                base, _, leaf = k.rpartition(".")
                if leaf in ("weight_packed", "weight_scale", "weight_global_scale",
                            "input_global_scale", "weight_shape"):
                    per_layer[base][leaf] = (shape, dt, b)
                    continue
                if leaf == "bias" and base in per_layer:
                    per_layer[base][leaf] = (shape, dt, b)
                    continue
                # all other tensors (unquantized weights, norms, embeddings, vision tower)
                total_params_all += n
    q_bytes = 0
    q_weights = 0
    n_layers = 0
    for base, d in per_layer.items():
        if "weight_packed" not in d:
            continue
        n_layers += 1
        shape, dt, b = d["weight_packed"]
        rows, cols_packed = shape
        nw = rows * cols_packed * 2
        q_weights += nw
        q_bytes += sum(v[2] for kk, v in d.items() if kk in
                       ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale"))
    total_params_all += q_weights
    out = {
        "dir": a.dir,
        "n_quantized_linears": n_layers,
        "quantized_weights": q_weights,
        "quantized_bytes": q_bytes,
        "bpw_quantized_linears": 8.0 * q_bytes / max(q_weights, 1),
        "checkpoint_bytes": total_bytes,
        "checkpoint_params_all": total_params_all,
        "bpw_whole_checkpoint": 8.0 * total_bytes / max(total_params_all, 1),
        "formula": "bpw = 8*bytes(weight_packed+weight_scale+weight_global_scale+input_global_scale)"
                   "/ (rows*packed_cols*2)",
    }
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
