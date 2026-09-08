#!/usr/bin/env python
"""Effective bits-per-weight of a compressed-tensors W4A16 artifact.

bpw_quantized_linears = 8 * sum_bytes(weight_packed + weight_scale + weight_zero_point) / sum(prod(weight_shape))
   over every module that has a `weight_packed` tensor (i.e. the quantized Linear layers).
bpw_all = 8 * (all safetensors bytes) / (all parameters, quantized + bf16 leftovers such as embeddings/norms)
"""
import glob
import json
import os
import sys

from safetensors import safe_open

out = sys.argv[1]
files = sorted(glob.glob(os.path.join(out, "*.safetensors")))
q_bytes = q_params = 0
all_bytes = all_params = 0
n_q = 0
dtype_bytes = {"I32": 4, "I8": 1, "U8": 1, "BF16": 2, "F16": 2, "F32": 4, "I64": 8, "I16": 2, "F8_E4M3": 1}
shapes = {}
per = {}
for fn in files:
    with safe_open(fn, "pt") as f:
        for k in f.keys():
            sl = f.get_slice(k)
            shp, dt = sl.get_shape(), sl.get_dtype()
            n = 1
            for s in shp:
                n *= s
            b = n * dtype_bytes[dt]
            all_bytes += b
            if k.endswith(".weight_shape"):
                shapes[k[: -len(".weight_shape")]] = list(f.get_tensor(k).tolist())
                continue
            base, _, leaf = k.rpartition(".")
            if leaf in ("weight_packed", "weight_scale", "weight_zero_point"):
                per.setdefault(base, 0)
                per[base] += b
            else:
                all_params += n
for base, b in per.items():
    r, c = shapes[base]
    q_params += r * c
    q_bytes += b
    n_q += 1
all_params += q_params
res = {
    "artifact": out,
    "n_quantized_linears": n_q,
    "quantized_params": q_params,
    "quantized_bytes": q_bytes,
    "bpw_quantized_linears": 8 * q_bytes / q_params,
    "total_params": all_params,
    "total_safetensor_bytes": all_bytes,
    "bpw_all_tensors": 8 * all_bytes / all_params,
    "formula": "bpw_q = 8*bytes(weight_packed+weight_scale+weight_zero_point)/prod(weight_shape) summed over quantized Linears; "
               "bpw_all = 8*all_safetensor_bytes/all_params",
}
print(json.dumps(res, indent=1))
