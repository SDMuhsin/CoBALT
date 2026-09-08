"""accel4bit: effective bits-per-weight of a compressed-tensors (llm-compressor) artifact.

Formula: for every quantized Linear (tensors `<prefix>.weight_packed`), bytes = sum of the safetensors byte
sizes of ALL tensors belonging to that Linear (`weight_packed`, `weight_scale`, `weight_zero_point`,
`weight_g_idx`, `weight_shape`, bias if present); n_params = out_features * in_features from `weight_shape`.
  bpw_quantized_linears = 8 * sum(bytes) / sum(n_params)          (embeddings / lm_head / norms EXCLUDED)
Also reported: bpw over ALL tensors of the checkpoint (embeddings + everything) for context.
Byte sizes come from the safetensors headers (dtype x shape), i.e. exactly what is stored on disk.
"""
import argparse
import glob
import json
import os
import struct

DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "I64": 8, "F64": 8,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U32": 4, "U16": 2, "I4": 0.5, "U4": 0.5}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--out-json")
    args = ap.parse_args()
    tensors = {}
    for f in sorted(glob.glob(os.path.join(args.artifact, "*.safetensors"))):
        for k, v in read_header(f).items():
            if k == "__metadata__":
                continue
            nbytes = v["data_offsets"][1] - v["data_offsets"][0]
            tensors[k] = dict(dtype=v["dtype"], shape=v["shape"], nbytes=nbytes)
    q_prefixes = sorted(k[: -len(".weight_packed")] for k in tensors if k.endswith(".weight_packed"))
    q_bytes = q_params = 0
    per = {}
    for p in q_prefixes:
        shape = tensors[p + ".weight_shape"]["shape"] if p + ".weight_shape" in tensors else None
        # weight_shape tensor holds [out, in]; read its value from the file is costly, use packed shape instead:
        packed = tensors[p + ".weight_packed"]["shape"]  # [out, in/8] int32 for 4-bit
        out_f = packed[0]
        in_f = packed[1] * 8 if tensors[p + ".weight_packed"]["dtype"] == "I32" else None
        n = out_f * in_f
        b = sum(t["nbytes"] for k, t in tensors.items() if k.startswith(p + ".") and k[len(p) + 1:].split(".")[0]
                in ("weight_packed", "weight_scale", "weight_zero_point", "weight_g_idx", "weight_shape", "bias"))
        comps = {k[len(p) + 1:]: (t["dtype"], t["shape"], t["nbytes"]) for k, t in tensors.items() if k.startswith(p + ".")}
        per[p] = dict(n_params=n, nbytes=b, bpw=8 * b / n, components=comps)
        q_bytes += b
        q_params += n
    all_bytes = sum(t["nbytes"] for t in tensors.values())
    # count all "weight-like" params: quantized linears + every non-quantized tensor's numel
    other_params = sum((1 if not t["shape"] else eval("*".join(map(str, t["shape"])))) for k, t in tensors.items()
                       if not any(k.startswith(p + ".") for p in q_prefixes))
    res = dict(artifact=args.artifact, n_quantized_linears=len(q_prefixes), quantized_linear_params=q_params,
               quantized_linear_bytes=q_bytes, bpw_quantized_linears=8 * q_bytes / q_params if q_params else None,
               all_tensor_bytes=all_bytes, all_params=q_params + other_params,
               bpw_all_tensors=8 * all_bytes / (q_params + other_params),
               example_components=per[q_prefixes[0]]["components"] if q_prefixes else None,
               unquantized_tensors=[k for k in tensors if not any(k.startswith(p + ".") for p in q_prefixes)][:40])
    print(json.dumps({k: v for k, v in res.items() if k not in ("example_components", "unquantized_tensors")}, indent=2))
    print("example components:", json.dumps(res["example_components"], indent=1))
    if args.out_json:
        json.dump(res, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
