"""Stage 4 -- verification.

Two levels, cheapest first.

`check_artifact()` -- seconds, no kernel build.  Reads the packed bytes back with the
torch dequantizer and asserts the things a packing bug breaks: every blob is long enough
for the arrays the manifest places in it, the bpw accounting adds up, every pruned
position decodes to an exact zero, and -- for the 16:32 layouts --
every aligned 32-column block of every row has exactly 16 survivors.

`check_kernel()` -- minutes, builds the CUDA extension.  Runs the megakernel against the
pure-torch reference forward pass on the SAME dequantized bytes, so a disagreement can
only be a kernel bug, never a quantization difference.

Why "exactly zero" is the assertion that matters: an asymmetric quantizer stores a
zero-point, and a pruned position decodes to `(0 - zero) * scale`, which is NOT zero
unless the packer put the grid's zero on a representable code.  Getting this wrong leaves
a model that still produces fluent text and quietly scores worse.
"""
from __future__ import annotations

import json
import os

from . import _bridge

# Matrices the torch oracle can reconstruct on their own. `qkv` is the row-fused
# q|k|v matrix the kernel actually streams; q_proj/k_proj/v_proj are ALIASES into it
# (`alias_of` in the manifest) and are the form with a one-row col_scale, so they are
# what gets inspected. Byte accounting uses the non-alias entries instead, or the fused
# matrix would be counted twice.
INSPECT = ("q_proj", "k_proj", "v_proj", "o_proj", "gateup", "down_proj")
BLOCK, BLOCK_KEEP = 32, 16


class VerificationError(AssertionError):
    pass


def _keep_mask(DQ, blob, entry, device):
    """The stored survivor mask, or None for layouts that do not carry one.

    Read from the bytes rather than inferred from `W != 0`: a survivor whose code equals
    the group's zero-point legitimately decodes to 0.0, so the dequantized values
    undercount survivors. DENSE4 stores no mask at all (it codes every position).
    """
    import torch
    lay = entry["layout"]
    if lay not in (DQ.LAYOUT_BLK1632_4, DQ.LAYOUT_BLK1632_6):
        return None
    K, N = entry["K"], entry["N"]
    stride = (N // 2) if lay == DQ.LAYOUT_BLK1632_6 else (N * 3 // 8)
    rowb = DQ._arr(blob, entry["arrays"]["data"], torch.uint8)[: K * stride].view(K, stride)
    sh8 = torch.arange(8, dtype=torch.uint8, device=device)
    return ((rowb[:, : N // 8].reshape(K, -1, 1) >> sh8) & 1).view(K, N).bool()


def _check(cond, msg, problems):
    if not cond:
        problems.append(msg)
    return cond


def check_artifact(artifact_dir: str, layers: int = 2, device: str = "cuda",
                   strict: bool = True) -> dict:
    """Structural + numerical check of a packed CBK1 artifact. Returns a report."""
    import torch
    _bridge.install()
    from cobaltkernel import dequant_ref as DQ

    man = json.load(open(os.path.join(artifact_dir, "manifest.json")))
    problems: list[str] = []
    rep = {"artifact": artifact_dir, "format": man.get("format"),
           "layout": man.get("layout"), "bpw": man["bpw"], "matrices": {},
           "problems": problems}

    _check(man.get("format") == "CBK1", f"format is {man.get('format')!r}, want 'CBK1'",
           problems)

    # ---- blob lengths cover every array the manifest places in them -------------
    total = 0
    for lay in man["layers"]:
        path = os.path.join(artifact_dir, lay["file"])
        if not _check(os.path.exists(path), f"missing blob {lay['file']}", problems):
            continue
        size = os.path.getsize(path)
        total += size
        for nm, e in lay["matrices"].items():
            for an, a in e["arrays"].items():
                _check(a["off"] + a["bytes"] <= size,
                       f"{lay['file']}:{nm}.{an} runs past the blob "
                       f"({a['off']}+{a['bytes']} > {size})", problems)
    rep["blob_bytes"] = total

    # ---- bpw accounting recomputes -------------------------------------------
    owned = [e for lay in man["layers"] for e in lay["matrices"].values()
             if "alias_of" not in e]
    params = sum(e["K"] * e["N"] for e in owned)
    stream = sum(e["stream_bytes"] for e in owned)
    if params:
        bpw = 8.0 * stream / params
        rep["bpw_recomputed"] = round(bpw, 6)
        _check(abs(bpw - man["bpw"]["bpw_packed"]) < 1e-6,
               f"bpw recompute {bpw:.6f} != manifest {man['bpw']['bpw_packed']:.6f}",
               problems)

    # ---- dequantize and inspect the mask --------------------------------------
    for lay in man["layers"][:layers]:
        blob = DQ._blob(os.path.join(artifact_dir, lay["file"]), device)
        for nm in INSPECT:
            e = lay["matrices"].get(nm)
            if e is None:
                continue
            W = DQ.dequant_matrix(blob, e, device, dtype=torch.float32)
            frac = float((W == 0).float().mean())
            info = {"layout": e["layout_name"], "shape": list(W.shape),
                    "zero_fraction": round(frac, 6),
                    "finite": bool(torch.isfinite(W).all())}
            _check(info["finite"], f"{lay['file']}:{nm} has non-finite weights", problems)
            # At least half of every matrix is pruned to an exact zero. MORE than half can
            # legitimately be zero: a survivor whose code equals its group's zero-point
            # decodes to 0.0 too (BioMistral-7B layer-0 q_proj reads 0.60). The survivor
            # count itself is asserted from the stored mask below, not from W != 0.
            _check(frac >= 0.5 - 0.02,
                   f"{lay['file']}:{nm} zero fraction {frac:.4f} is below 0.50", problems)

            keep = _keep_mask(DQ, blob, e, device)
            if keep is not None:
                K, N = W.shape
                per_block = keep.reshape(K, N // BLOCK, BLOCK).sum(-1)
                lo, hi = int(per_block.min()), int(per_block.max())
                info["block_survivors"] = [lo, hi]
                _check(lo == hi == BLOCK_KEEP,
                       f"{lay['file']}:{nm} has {lo}..{hi} survivors per 32-block, want "
                       f"exactly {BLOCK_KEEP} -- the fixed-offset decoder relies on it",
                       problems)
                # THE assertion. A pruned position decodes as (0 - zero) * scale, which
                # is only zero if the packer put the grid's zero on a representable code.
                n_bad = int((W[~keep] != 0).sum())
                info["pruned_nonzero"] = n_bad
                info["survivors_on_zero_code"] = int((W[keep] == 0).sum())
                _check(n_bad == 0,
                       f"{lay['file']}:{nm} has {n_bad} pruned positions that do NOT "
                       "decode to exactly zero (zero-point off the quantization grid)",
                       problems)
            rep["matrices"][f"{lay['file']}:{nm}"] = info
        del blob

    rep["ok"] = not problems
    if problems and strict:
        raise VerificationError("\n  ".join(["artifact check FAILED:"] + problems))
    return rep


# (iii) on the tensor-core decode build: M=1 runs qkv/o_proj on mma with fp32 slice sums,
# M>1 runs the row loop, so the two are different correct reduction orders. The gate is
# then identical greedy tokens plus this bound on max |logit diff| (measured 0.09-0.14 on
# BioMistral-7B; the same-code GEMM-vs-GEMV reference control is 0.97).
BATCH_TOL_TENSOR_CORE = 0.5


def check_kernel(artifact_dir: str, config_dir: str | None = None, prompt_len: int = 1152,
                 steps: int = 32, recipe=None, out: str | None = None) -> None:
    """Run the megakernel against the pure-torch reference on the same bytes.

    Reports, per position, max|logit diff| and argmax agreement, then a greedy
    token-for-token comparison, then a batch-vs-single bit-exactness check.  The bar is
    the bf16 HF-eager-vs-HF-sdpa floor (98.5% argmax agreement) -- two mathematically
    identical bf16 implementations disagree that much on their own.

    Keep `prompt_len` at its default: the bar was established over 1152 positions, and at
    128 positions a single disagreement moves the figure by 0.8 pt.
    """
    from . import env as _env, recipes as _recipes, model as _model
    _env.configure()
    r = (_recipes.get(recipe) if isinstance(recipe, str)
         else recipe or _model.recipe_for_artifact(artifact_dir))
    # --ref-packed builds the torch oracle by dequantizing the SAME packed bytes the
    # kernel reads, so any disagreement is a kernel bug and not a quantization difference.
    argv = ["--model", artifact_dir, "--config", config_dir or artifact_dir, "--ref-packed",
            "--prompt-len", prompt_len, "--steps", steps]
    if out:
        argv += ["--out", out]
    with _recipes.activate(r, _model._model_cfg(config_dir or artifact_dir)):
        if os.environ.get("COBALT_BLK1632_MMA", "0") != "0":
            argv += ["--batch-tol", BATCH_TOL_TENSOR_CORE]
        _bridge.run_module_main("cobaltkernel.verify_kernel", argv)
