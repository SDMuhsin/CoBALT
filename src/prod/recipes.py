"""The shipped CoBALT arms, pinned end to end.

Each `Recipe` carries the *complete* configuration of one deployable arm -- quantizer
flags, packer flags, kernel build/runtime environment -- plus the numbers we measured
for it, so a partner can tell reproduction from regression without reading a log.

Why this file exists
--------------------
The kernel has ~25 environment knobs.  Several of them (`COBALT_BLK1632_NOOVH`,
`COBALT_ATTN_DIAG`, `COBALT_BLK1632_HALFJ`, `COBALT_BLK1632_NOLD`, `COBALT_XSYNC`,
`COBALT_BLK1632_NOXB`) are *diagnostics that deliberately compute the wrong answer* --
they exist to price a lever by deleting work.  A stray export of any of them produces a
fast model that generates nonsense, and nothing in the kernel complains.  Every entry
point in `prod` therefore runs inside `activate()`, which sets the pinned knobs and
**unsets every diagnostic**.  There is no supported way to reach a diagnostic build
through this package; use `src/cobaltkernel/` directly if you want one.

    from prod import recipes
    r = recipes.get("blk1632_b4")
    with recipes.activate(r):
        ...                      # kernel builds and runs in r's pinned configuration
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
from typing import Any

# Knobs that change the NUMERICS to something known-wrong. Cleared unconditionally.
DIAGNOSTIC_KNOBS = (
    "COBALT_BLK1632_NOXB",      # drops the 2nd gate/up activation stream
    "COBALT_BLK1632_HALFJ",     # halves the (extract, cvt, FMA) triples
    "COBALT_BLK1632_NOOVH",     # deletes the per-granule selector/mask overhead
    "COBALT_BLK1632_NOLD",      # skips the mask and/or nibble loads
    "COBALT_XSYNC",             # injects empty grid.sync barriers
    "COBALT_ATTN_DIAG",         # skips q.k and/or p.V
    "COBALT_BLK1632_ZFILL",     # test-harness zero-fill probe
    "COBALT_LUT_FAKE",          # test-harness LUT probe
)

# Knobs that are legitimate but arm-specific; cleared so one recipe never inherits
# another's settings inside the same process.
ARM_KNOBS = (
    "COBALT_BLK1632", "COBALT_BLK1632_O", "COBALT_BLK1632_LUT", "COBALT_BLK1632_ZF",
    "COBALT_BLK1632_BIAS", "COBALT_BLK1632_FT", "COBALT_BLK1632_PFX",
    "COBALT_BLK1632_MSCHED", "COBALT_GATEUP_RP", "COBALT_GATEUP_SPLIT",
    "COBALT_KVONCE", "COBALT_KPB", "COBALT_MINB", "COBALT_BLOCKS", "COBALT_FUSEPREP",
    "COBALT_M1ONLY", "COBALT_PROF", "COBALT_PF_BLK_LUT", "COBALT_PF_TAILSPLIT",
    "COBALT_PF_BATCH_MINB",
)


@dataclasses.dataclass(frozen=True)
class Recipe:
    """One deployable arm: how to build it, how to run it, what it should score."""
    name: str
    summary: str
    layout: str                     # CBK1 packing layout of the six/seven linears
    bpw: float                      # effective bits per weight of the packed artifact
    quantizer: dict[str, Any]       # -> prod.quantize.run()
    packer: dict[str, Any]          # -> prod.pack.run()
    kernel_env: dict[str, str]      # pinned build/runtime knobs
    reference: dict[str, Any]       # what we measured (see `reference_note`)
    notes: str = ""

    def env(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.kernel_env.items()}


# What every `reference` block below was measured on. Absolute tok/s will differ on
# other hardware; the RATIOS between arms are the reproducible part.
REFERENCE_NOTE = (
    "MedGemma-27B-text-it, one NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb slice "
    "(94 SM, 770.8 GB/s measured read bandwidth), CUDA 13.0 / torch 2.13, "
    "512-token prompt -> 128 generated tokens, greedy, batch 1. Quality: lm_eval "
    "0.4.13, 0-shot, seed 1234, medmcqa limited to 1000 items, run on the "
    "fake-quantised bf16 checkpoint under vLLM. 'x Q4_K_M' is against llama.cpp "
    "Q4_K_M (40.10 tok/s) measured on the same slice with the same protocol."
)

# The model every `reference` block was measured on. `compare_to_reference` refuses to
# quote a ratio when the benchmarked model is a different one -- a 4B smoke run against a
# 27B reference produces a flattering number that means nothing.
REFERENCE_MODEL = {"name": "MedGemma-27B-text-it", "layers": 62, "hidden": 5376}

_CALIB = "results/accel4bit/calib_ultrachat_512x2048.txt"

_BASE_QUANT = dict(sparsity=0.5, beta=0.5, group_size=128, hull="survivor",
                   calib="ultrachat", calib_file=_CALIB, n_calib=128, seq_len=2048)

_BASE_PACK = dict(embed_layout="DENSE4", embed_bits=4, fuse_qkv=True)


RECIPES: dict[str, Recipe] = {}


def _add(r: Recipe) -> Recipe:
    RECIPES[r.name] = r
    return r


_add(Recipe(
    name="dense4",
    summary="CoBALT 50% unstructured mask, 4-bit survivors, stored DENSE (a code for "
            "every position, pruned ones exactly zero). The control.",
    layout="DENSE4",
    bpw=4.1875,
    quantizer=dict(_BASE_QUANT, bits=4, mask_block=0),
    packer=dict(_BASE_PACK, layout="DENSE4", bits=4),
    # MEASURED: the DENSE4 control prefers a 256-key attention split (+0.55%); the
    # BLK arms are flat on this axis and ship at the default 128.
    kernel_env={"COBALT_KPB": 256},
    reference=dict(decode_tok_s=41.75, x_q4km=1.041, ttft_ms=314.3, prefill_tok_s=1629,
                   artifact_gb=14.151, peak_mib_m8=16534, med_avg=62.47, wiki_ppl=15.933,
                   arc_easy=0.8106),
    notes="The mask buys no bytes here: at 50% unstructured sparsity the variable-length "
          "bitmap formats are decode-bound below dense streaming, so this arm stores the "
          "zeros. It is the speed and quality control every other arm is measured against.",
))

_add(Recipe(
    name="blk1632_b4",
    summary="CoBALT-16:32 fixed-cardinality block mask (exactly 16 survivors per aligned "
            "32-column block), 4-bit survivors. FASTEST arm.",
    layout="BLK1632_4",
    bpw=3.1875,
    quantizer=dict(_BASE_QUANT, bits=4, mask_block=32),
    packer=dict(_BASE_PACK, layout="BLK1632_4", bits=4),
    kernel_env={"COBALT_BLK1632": 4},
    reference=dict(decode_tok_s=46.92, x_q4km=1.170, ttft_ms=328.6, prefill_tok_s=1558,
                   artifact_gb=10.951, peak_mib_m8=13440, med_avg=62.25, wiki_ppl=16.842,
                   arc_easy=0.8068),
    notes="Fixed cardinality is what makes the sparsity payable: survivor positions land "
          "at compile-time-known offsets, so the decoder needs no prefix popcount and no "
          "sliding shift chain -- 24% fewer bytes per token at a 0.28 pt task-average cost.",
))

_add(Recipe(
    name="blk1632_b4_oproj",
    summary="blk1632_b4 with o_proj left on the canonical global-top-k mask (DENSE4). "
            "BEST QUALITY PER BYTE.",
    layout="BLK1632_4 + DENSE4 o_proj",
    bpw=3.2408,
    quantizer=dict(_BASE_QUANT, bits=4, mask_block=32,
                   mask_block_exclude="self_attn.o_proj"),
    packer=dict(_BASE_PACK, layout="BLK1632_4", bits=4, layout_override="o_proj=DENSE4"),
    kernel_env={"COBALT_BLK1632": 4, "COBALT_BLK1632_O": 0},
    reference=dict(decode_tok_s=46.74, x_q4km=1.166, ttft_ms=330.2, prefill_tok_s=1551,
                   artifact_gb=11.122, peak_mib_m8=13564, med_avg=63.59, wiki_ppl=16.550,
                   arc_easy=0.8081),
    notes="o_proj is the one matrix the block constraint really hurts: its output error "
          "rises 42% under 16:32 versus 9-10% everywhere else. Exempting it costs 0.05 bpw "
          "and 0.2 tok/s and buys 1.34 pt of medical task average. The damage is the MASK, "
          "not the bit width -- which is why the exemption, not more bits, is the fix.",
))

_add(Recipe(
    name="blk1632_b6_oproj4",
    summary="16:32 mask with 6-bit survivors, o_proj DENSE4 at 4-bit: exactly the same "
            "bytes as dense4 (4.1875 bpw), higher quality. BEST QUALITY AT MATCHED MEMORY.",
    layout="BLK1632_6 + DENSE4 o_proj",
    bpw=4.1875,
    quantizer=dict(_BASE_QUANT, bits=6, mask_block=32,
                   mask_block_exclude="self_attn.o_proj",
                   bits_override="self_attn.o_proj=4"),
    packer=dict(_BASE_PACK, layout="BLK1632_6", bits=6,
                layout_override="o_proj=DENSE4", bits_override="o_proj=4"),
    kernel_env={"COBALT_BLK1632": 6, "COBALT_BLK1632_O": 0},
    reference=dict(decode_tok_s=37.18, x_q4km=0.927, ttft_ms=367.7, prefill_tok_s=1392,
                   artifact_gb=14.151, peak_mib_m8=16540, med_avg=64.39, wiki_ppl=16.291,
                   arc_easy=0.8102),
    notes="Packed byte count is the same INTEGER as dense4 (13,399,142,400 B) and the "
          "medical task average is 1.92 pt higher -- but decode is 11% slower, because a "
          "6-bit survivor plane costs an extra load per granule that the byte count hides. "
          "Choose this arm when quality per byte matters more than latency.",
))


def names() -> list[str]:
    return list(RECIPES)


def get(name: str) -> Recipe:
    try:
        return RECIPES[name]
    except KeyError:
        raise KeyError(f"unknown recipe {name!r}; known: {', '.join(RECIPES)}") from None


def describe(name: str | None = None) -> str:
    rs = [get(name)] if name else list(RECIPES.values())
    out = []
    for r in rs:
        out.append(f"{r.name}  [{r.layout}, {r.bpw:.4f} bpw]\n  {r.summary}")
        ref = r.reference
        out.append(f"  measured: {ref['decode_tok_s']} tok/s ({ref['x_q4km']}x Q4_K_M), "
                   f"{ref['artifact_gb']} GB, medical avg {ref['med_avg']}")
        if r.kernel_env:
            out.append("  kernel env: " + " ".join(f"{k}={v}" for k, v in r.env().items()))
        if r.notes:
            out.append("  " + r.notes.replace("\n", "\n  "))
        out.append("")
    return "\n".join(out)


@contextlib.contextmanager
def activate(recipe: Recipe | str):
    """Run a block with `recipe`'s kernel configuration and NO diagnostic knob set.

    Restores the previous environment on exit, so several arms can be exercised in one
    process (each still pays its own JIT build -- the knobs are compile-time macros).
    """
    r = get(recipe) if isinstance(recipe, str) else recipe
    saved = {k: os.environ.get(k) for k in (*DIAGNOSTIC_KNOBS, *ARM_KNOBS)}
    try:
        for k in (*DIAGNOSTIC_KNOBS, *ARM_KNOBS):
            os.environ.pop(k, None)
        os.environ.update(r.env())
        yield r
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
