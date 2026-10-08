#!/usr/bin/env python
"""One-time conversion of a single-file `pytorch_model.bin` HF checkpoint into a sharded
bf16 safetensors snapshot (config + tokenizer files copied), so that every consumer in
this tree -- the streaming quantizer's ShardReader, vLLM, llama.cpp's converter -- reads
the same bytes from one directory.

  python convert_bin_to_safetensors.py --src <hf snapshot with pytorch_model.bin> --out <dir>
"""
import argparse, json, os, shutil, time

import torch
from safetensors.torch import save_file

COPY = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "chat_template.jinja"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gb", type=float, default=4.0)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "keep"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    for f in COPY:
        s = os.path.join(a.src, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(a.out, f))
    sd = torch.load(os.path.join(a.src, "pytorch_model.bin"), map_location="cpu",
                    weights_only=True, mmap=True)
    dtypes = {}
    for k, v in sd.items():
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + 1
    print(f"[convert] loaded {len(sd)} tensors in {time.time()-t0:.0f}s, source dtypes {dtypes}", flush=True)
    tgt = None if a.dtype == "keep" else getattr(torch, a.dtype)

    # shard by cumulative bytes, in checkpoint key order
    limit = int(a.shard_gb * 1e9)
    shards, cur, cur_b = [], {}, 0
    for k, v in sd.items():
        t = v if tgt is None or not v.is_floating_point() else v.to(tgt)
        t = t.contiguous()
        if cur and cur_b + t.numel() * t.element_size() > limit:
            shards.append(cur); cur, cur_b = {}, 0
        cur[k] = t; cur_b += t.numel() * t.element_size()
    if cur:
        shards.append(cur)
    n = len(shards)
    wmap, total = {}, 0
    for i, sh in enumerate(shards):
        name = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        save_file(sh, os.path.join(a.out, name), metadata={"format": "pt"})
        for k, t in sh.items():
            wmap[k] = name; total += t.numel() * t.element_size()
        print(f"[convert] wrote {name} ({len(sh)} tensors)", flush=True)
    json.dump({"metadata": {"total_size": total}, "weight_map": wmap},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
    # record provenance + make the config's dtype honest
    cfg_p = os.path.join(a.out, "config.json")
    cfg = json.load(open(cfg_p))
    if tgt is not None:
        cfg["torch_dtype"] = a.dtype
    json.dump(cfg, open(cfg_p, "w"), indent=2)
    json.dump({"source": a.src, "source_dtypes": dtypes, "target_dtype": a.dtype,
               "n_tensors": len(wmap), "total_bytes": total, "shards": n,
               "wall_s": time.time() - t0},
              open(os.path.join(a.out, "CONVERSION.json"), "w"), indent=1)
    print(f"DONE {len(wmap)} tensors, {total/1e9:.2f} GB in {n} shards, {time.time()-t0:.0f}s -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
