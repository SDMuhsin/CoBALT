"""Stage 2 -- raw CoBALT artifact -> CBK1 packed artifact (the VRAM image).

The packed directory is a byte-for-byte image of what the kernel reads: loading a model
is `open()` + `cudaMemcpy`, with no unpacking step and no host-side reshuffle, so disk
bytes == VRAM bytes.  `manifest.json` describes every array's offset and length.

Layouts (see docs/FORMAT.md):
  DENSE4       a 4-bit code for every position; pruned positions hold an exact zero
  BLK1632_4    16:32 block mask, 4-bit survivors -- 3.1875 bpw
  BLK1632_6    16:32 block mask, 6-bit survivors -- 4.1875 bpw (== DENSE4 bytes)

    from prod import pack, recipes
    pack.run(recipes.get("blk1632_b4"), raw=..., out=..., model_path=<hf snapshot>)

`model_path` is needed because the tied embedding / lm_head and the RMSNorm weights come
straight from the HF snapshot; the quantizer only writes the seven decoder linears.
"""
from __future__ import annotations

import json
import os

from . import _bridge, recipes as _recipes


def build_argv(recipe, raw: str, out: str, model_path: str | None,
               device: str = "cuda", extra: list[str] | None = None) -> list[str]:
    p = recipe.packer
    argv = ["--raw", raw, "--out", out, "--layout", p["layout"], "--bits", p["bits"],
            "--embed-layout", p["embed_layout"], "--embed-bits", p["embed_bits"],
            "--device", device]
    if p.get("fuse_qkv"):
        argv += ["--fuse-qkv"]
    if p.get("layout_override"):
        argv += ["--layout-override", p["layout_override"]]
    if p.get("bits_override"):
        argv += ["--bits-override", p["bits_override"]]
    if model_path:
        argv += ["--model-path", model_path]
    return argv + list(extra or [])


def run(recipe, raw: str, out: str, model_path: str | None = None,
        device: str = "cuda", extra: list[str] | None = None) -> dict:
    """Pack `raw` into `out`. Returns the manifest's bpw accounting block."""
    r = _recipes.get(recipe) if isinstance(recipe, str) else recipe
    os.makedirs(out, exist_ok=True)
    _bridge.run_module_main("pack_cobalt",
                            build_argv(r, raw, out, model_path, device, extra))
    man = json.load(open(os.path.join(out, "manifest.json")))
    return man["bpw"]
