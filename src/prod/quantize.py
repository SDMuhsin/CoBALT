"""Stage 1 -- HF checkpoint -> raw CoBALT artifact.

CoBALT is joint pruning + quantization applied layer by layer:

  1. Wanda importance          |W_ij| * ||X_j||_2   from a calibration pass
  2. row + column quantile self-normalisation (the "column balance", exponent `beta`)
  3. selection: one global top-k over the matrix, OR -- with `mask_block=32` -- exactly
     16 survivors in every aligned block of 32 input columns
  4. OBS compensation of the pruned mass into the survivors, H = X^T X + lambda*I
  5. sparse-aware per-column scale, then group-128 asymmetric RTN over the survivors

The model is never materialised: decoder layers are built on `meta` and their weights
streamed from the snapshot one layer at a time, so a 27B model quantizes inside a 48 GB
slice.  Expect hours, not minutes -- launch it detached.

    from prod import quantize, recipes
    quantize.run(recipes.get("blk1632_b4"), model_path=..., out=...)
"""
from __future__ import annotations

import os

from . import _bridge, recipes as _recipes


def build_argv(recipe, model_path: str, out: str, device: str = "cuda",
               extra: list[str] | None = None) -> list[str]:
    q = recipe.quantizer
    argv = ["--model-path", model_path, "--out", out, "--device", device,
            "--sparsity", q["sparsity"], "--bits", q["bits"], "--beta", q["beta"],
            "--group-size", q["group_size"], "--hull", q["hull"],
            "--calib", q["calib"], "--calib-file", _bridge.repo_path(q["calib_file"]),
            "--n-calib", q["n_calib"], "--seq-len", q["seq_len"]]
    if q.get("mask_block"):
        argv += ["--mask-block", q["mask_block"]]
    if q.get("mask_block_exclude"):
        argv += ["--mask-block-exclude", q["mask_block_exclude"]]
    if q.get("bits_override"):
        argv += ["--bits-override", q["bits_override"]]
    return argv + list(extra or [])


def run(recipe, model_path: str, out: str, device: str = "cuda",
        extra: list[str] | None = None) -> str:
    """Quantize `model_path` into the raw artifact directory `out`. Returns `out`."""
    r = _recipes.get(recipe) if isinstance(recipe, str) else recipe
    os.makedirs(out, exist_ok=True)
    _bridge.run_module_main("quantize_cobalt",
                            build_argv(r, model_path, out, device, extra))
    return out
