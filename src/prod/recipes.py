"""The shipped CoBALT arms and model targets, pinned end to end.

Two tables live here.

`RECIPES` -- one `Recipe` per deployable ARM: quantizer flags, packer flags and the
kernel knobs that belong to the layout.  A recipe is model-independent: the same
`blk1632_b4` recipe built the MedGemma-27B artifact and the BioMistral-7B artifact.

`TARGETS` -- one `Target` per shipped MODEL: the per-model kernel tuning that sits on top
of the recipe's knobs, and the numbers we measured for every (model, recipe) pair, so a
partner can tell reproduction from regression without reading a log.  A target is picked
from the artifact's own `config.json` (`model_type`, `hidden_size`), never by hand.

Why this file exists
--------------------
The kernel has ~40 environment knobs.  Several of them (`COBALT_BLK1632_NOOVH`,
`COBALT_ATTN_DIAG`, `COBALT_BLK1632_HALFJ`, `COBALT_BLK1632_NOLD`, `COBALT_XSYNC`,
`COBALT_BLK1632_NOXB`, ...) are *diagnostics that deliberately compute the wrong answer* --
they exist to price a lever by deleting work.  A stray export of any of them produces a
fast model that generates nonsense, and nothing in the kernel complains.  Every entry
point in `prod` therefore runs inside `activate()`, which sets the pinned knobs and
**unsets every diagnostic**.  There is no supported way to reach a diagnostic build
through this package; use `src/cobaltkernel/` directly if you want one.

    from prod import recipes
    r = recipes.get("blk1632_b4")
    with recipes.activate(r, model_cfg):
        ...                      # kernel builds and runs in r's pinned configuration,
                                 # plus the target's tuning for this model
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
    "COBALT_BLK1632_LOADONLY",  # streams the weights, skips the decode
    "COBALT_MMA_NOLD",          # tensor-core path without its weight loads
    "COBALT_XSYNC",             # injects empty grid.sync barriers
    "COBALT_ATTN_DIAG",         # skips q.k and/or p.V
    "COBALT_SMEM_MIN",          # inflates the dynamic smem (prices the L1/smem carve-out)
    "COBALT_BLK1632_ZFILL",     # test-harness zero-fill probe
    "COBALT_LUT_FAKE",          # test-harness LUT probe
)

# Knobs that are legitimate but arm- or model-specific; cleared so one recipe never
# inherits another's settings inside the same process.
ARM_KNOBS = (
    "COBALT_BLK1632", "COBALT_BLK1632_O", "COBALT_BLK1632_LUT", "COBALT_BLK1632_ZF",
    "COBALT_BLK1632_BIAS", "COBALT_BLK1632_FT", "COBALT_BLK1632_PFX",
    "COBALT_BLK1632_MSCHED", "COBALT_GATEUP_RP", "COBALT_GATEUP_SPLIT",
    "COBALT_KVONCE", "COBALT_KPB", "COBALT_MINB", "COBALT_BLOCKS", "COBALT_FUSEPREP",
    "COBALT_M1ONLY", "COBALT_PROF", "COBALT_PF_BLK_LUT", "COBALT_PF_TAILSPLIT",
    "COBALT_PF_BATCH_MINB",
    # model-family port: query:kv ratio build, attention register pressure, GEMV schedule
    "COBALT_KG", "COBALT_ATTN_KUNROLL", "COBALT_PVB", "COBALT_RMAX", "COBALT_CSPLIT",
    "COBALT_FUSE_RESID", "COBALT_ATTN_COMBINE1", "COBALT_PTXAS_V",
    # tensor-core 16:32 decode on the L2-resident phases
    "COBALT_BLK1632_MMA", "COBALT_MMA_PHASES", "COBALT_MMA_PF", "COBALT_MMA_PF_L2",
    "COBALT_MMA_UPW", "COBALT_MMA_UPW_O", "COBALT_MMA_ACC2", "COBALT_MMA_COAL", "COBALT_XSMEM",
    "COBALT_SZPRE",
    # tensor-core decode attention, selector+mask table, row-decoder lookahead, and the
    # measured-negative variants kept in-tree (chunked / cp.async-ring walks, tree merge,
    # q-RoPE in fragments)
    "COBALT_ATTN_TC", "COBALT_ATTN_TC_REDUCE", "COBALT_ATTN_TC_MOVM", "COBALT_ATTN_TC_HALF",
    "COBALT_ATTN_TC_TREE", "COBALT_ATTN_TC_QROPE", "COBALT_ATTN_TC_VPRE", "COBALT_BLK1632_PFH",
    "COBALT_MMA_SZPRE", "COBALT_CHUNK", "COBALT_STREAM", "COBALT_STREAM_S", "COBALT_STREAM_PF",
)


# ============================================================================ recipes
@dataclasses.dataclass(frozen=True)
class Recipe:
    """One deployable arm: how to build it and how to run it (model-independent)."""
    name: str
    summary: str
    layout: str                     # CBK1 packing layout of the six/seven linears
    bpw: float                      # effective bits per weight of the packed artifact
    quantizer: dict[str, Any]       # -> prod.quantize.run()
    packer: dict[str, Any]          # -> prod.pack.run()
    kernel_env: dict[str, str]      # pinned build/runtime knobs that belong to the layout
    notes: str = ""

    def env(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.kernel_env.items()}


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
    notes="The mask buys no bytes here: at 50% unstructured sparsity the variable-length "
          "bitmap formats are decode-bound below dense streaming, so this arm stores the "
          "zeros. It is the speed and quality control every other arm is measured against.",
))

_add(Recipe(
    name="blk1632_b4",
    summary="CoBALT-16:32 fixed-cardinality block mask (exactly 16 survivors per aligned "
            "32-column block), 4-bit survivors. FASTEST arm; the one shipped for every model.",
    layout="BLK1632_4",
    bpw=3.1875,
    quantizer=dict(_BASE_QUANT, bits=4, mask_block=32),
    packer=dict(_BASE_PACK, layout="BLK1632_4", bits=4),
    kernel_env={"COBALT_BLK1632": 4},
    notes="Fixed cardinality is what makes the sparsity payable: survivor positions land "
          "at compile-time-known offsets, so the decoder needs no prefix popcount and no "
          "sliding shift chain -- 24% fewer bytes per token than DENSE4.",
))

_add(Recipe(
    name="blk1632_b4_oproj",
    summary="blk1632_b4 with o_proj left on the canonical global-top-k mask (DENSE4). "
            "BEST QUALITY PER BYTE on MedGemma-27B.",
    layout="BLK1632_4 + DENSE4 o_proj",
    bpw=3.2408,
    quantizer=dict(_BASE_QUANT, bits=4, mask_block=32,
                   mask_block_exclude="self_attn.o_proj"),
    packer=dict(_BASE_PACK, layout="BLK1632_4", bits=4, layout_override="o_proj=DENSE4"),
    kernel_env={"COBALT_BLK1632": 4, "COBALT_BLK1632_O": 0},
    notes="o_proj is the one matrix the block constraint really hurts on MedGemma: its "
          "output error rises 42% under 16:32 versus 9-10% everywhere else. Exempting it "
          "costs 0.05 bpw and 0.2 tok/s and buys 1.34 pt of medical task average. The "
          "damage is the MASK, not the bit width -- which is why the exemption, not more "
          "bits, is the fix.",
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
    notes="Packed byte count is the same INTEGER as dense4 and the medical task average is "
          "higher -- but decode is 11% slower, because a 6-bit survivor plane costs an "
          "extra load per granule that the byte count hides. Choose this arm when quality "
          "per byte matters more than latency.",
))


# ============================================================================ targets
@dataclasses.dataclass(frozen=True)
class Target:
    """One shipped model: its kernel tuning and what each recipe measured on it."""
    name: str                       # artifact directory prefix, e.g. "biomistral-7b"
    model_type: str                 # config.json model_type
    hidden: int                     # config.json hidden_size (the two together are the key)
    layers: int
    summary: str
    kernel_env: dict[str, str]      # per-model tuning applied on top of the recipe's knobs
    reference_note: str             # what every `reference` block was measured on
    reference: dict[str, dict]      # recipe name -> measured numbers
    baseline: dict[str, Any]        # llama.cpp Q4_K_M on the same slice, same protocol

    @property
    def key(self) -> tuple[str, int]:
        return (self.model_type, self.hidden)

    def env(self) -> dict[str, str]:
        return {k: str(v) for k, v in self.kernel_env.items()}


TARGETS: dict[str, Target] = {}


def _add_target(t: Target) -> Target:
    TARGETS[t.name] = t
    return t


_QUALITY_PROTOCOL = ("Quality: lm_eval 0.4.13, 0-shot, seed 1234, medmcqa limited to 1000 "
                     "items, run on the fake-quantised bf16 checkpoint under vLLM; "
                     "med_avg = mean(MedQA, PubMedQA, MedMCQA) x 100.")

_add_target(Target(
    name="medgemma-27b",
    model_type="gemma3_text", hidden=5376, layers=62,
    summary="MedGemma-27B-text-it (Gemma3 architecture). Four arms shipped.",
    kernel_env={},                  # the recipes' own knobs are the tuned configuration
    reference_note=(
        "MedGemma-27B-text-it, one NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb slice "
        "(94 SM, 770.8 GB/s measured read bandwidth), CUDA 13.0 / torch 2.13, "
        "512-token prompt -> 128 generated tokens, greedy, batch 1. " + _QUALITY_PROTOCOL +
        " 'x Q4_K_M' is against llama.cpp Q4_K_M (40.10 tok/s) measured on the same slice "
        "with the same protocol."),
    baseline=dict(decode_tok_s=40.10, artifact_gb=16.55, ttft_ms=237, med_avg=60.14,
                  wiki_ppl=11.6),
    reference={
        "dense4": dict(decode_tok_s=41.75, x_q4km=1.041, ttft_ms=314.3, prefill_tok_s=1629,
                       artifact_gb=14.151, peak_mib_m8=16534, med_avg=62.47,
                       wiki_ppl=15.933, arc_easy=0.8106),
        "blk1632_b4": dict(decode_tok_s=46.92, x_q4km=1.170, ttft_ms=328.6,
                           prefill_tok_s=1558, artifact_gb=10.951, peak_mib_m8=13440,
                           med_avg=62.25, wiki_ppl=16.842, arc_easy=0.8068),
        "blk1632_b4_oproj": dict(decode_tok_s=46.74, x_q4km=1.166, ttft_ms=330.2,
                                 prefill_tok_s=1551, artifact_gb=11.122, peak_mib_m8=13564,
                                 med_avg=63.59, wiki_ppl=16.550, arc_easy=0.8081),
        "blk1632_b6_oproj4": dict(decode_tok_s=37.18, x_q4km=0.927, ttft_ms=367.7,
                                  prefill_tok_s=1392, artifact_gb=14.151, peak_mib_m8=16540,
                                  med_avg=64.39, wiki_ppl=16.291, arc_easy=0.8102),
    },
))

_add_target(Target(
    name="biomistral-7b",
    model_type="mistral", hidden=4096, layers=32,
    summary="BioMistral-7B (Mistral-7B-v0.1 base, llama family). One arm shipped: "
            "blk1632_b4 on the tensor-core decode build.",
    # MEASURED on the 2g slice, GPU-side clock64 step; every alternative lost (see
    # docs/KERNELS.md s6). The KG=4 (32 query : 8 kv heads) build sits at the 128-register
    # bound at 2 blocks/SM; KUNROLL 4 and PVB 2 un-spill it.
    kernel_env={
        "COBALT_BLOCKS": "160", "COBALT_KPB": "128",
        "COBALT_ATTN_KUNROLL": "4", "COBALT_PVB": "2",
        # tensor-core decode of the L2-resident qkv / o_proj phases (M == 1 only):
        # 2g-slice step 7.20 -> 6.80 ms; the DRAM-streaming phases keep the row loop
        "COBALT_BLK1632_MMA": "1", "COBALT_MMA_PHASES": "3",
        "COBALT_MMA_PF_L2": "1", "COBALT_MMA_UPW": "2", "COBALT_MMA_UPW_O": "1",
        # tensor-core decode attention (mma.m16n8k16, fused cross-split reduce, fp16 merge
        # slots -- the dynamic smem must stay under the 32 KB L1/smem carve-out step), the
        # {sel0,sel1,mk0,mk1} selector table, and a one-iteration mask+nibble lookahead in
        # the row decoder at PF 1: step 6.84 -> 6.15 ms, 146 -> 164 tok/s in-kernel
        "COBALT_ATTN_TC": "1", "COBALT_BLK1632_LUT": "4",
        "COBALT_BLK1632_MSCHED": "3", "COBALT_BLK1632_PFH": "2",
    },
    reference_note=(
        "BioMistral-7B, one NVIDIA RTX PRO 6000 Blackwell MIG 2g.48gb slice (94 SM), "
        "CUDA 13.0 / torch 2.13, 512-token prompt -> 128 generated tokens, greedy, batch 1, "
        "host load < 4, llama.cpp and this kernel interleaved in one window and pinned to "
        "the same cores; numbers are the mean of two passes. " + _QUALITY_PROTOCOL +
        " 'x Q4_K_M' is against llama.cpp Q4_K_M with imatrix (143.75 tok/s, 4.37 GB) "
        "measured on the same slice with the same protocol."),
    baseline=dict(decode_tok_s=143.75, artifact_gb=4.37, ttft_ms=77.0, prefill_tok_s=6656,
                  med_avg=53.23, wiki_ppl=13.574, arc_easy=0.7811),
    reference={
        "blk1632_b4": dict(decode_tok_s=164.30, decode_tok_s_inkernel=164.75, x_q4km=1.143,
                           kernel_step_ms=6.11, ttft_ms=112.0, prefill_tok_s=4572,
                           artifact_gb=2.923, peak_mib_m8=3874, med_avg=49.72,
                           wiki_ppl=15.469, arc_easy=0.7567),
    },
))


# ============================================================================ lookup
def names() -> list[str]:
    return list(RECIPES)


def get(name: str) -> Recipe:
    try:
        return RECIPES[name]
    except KeyError:
        raise KeyError(f"unknown recipe {name!r}; known: {', '.join(RECIPES)}") from None


def target_names() -> list[str]:
    return list(TARGETS)


def get_target(name: str) -> Target:
    try:
        return TARGETS[name]
    except KeyError:
        raise KeyError(f"unknown target {name!r}; known: {', '.join(TARGETS)}") from None


def target_for(model_cfg: dict | None) -> Target | None:
    """The shipped target whose (model_type, hidden_size) matches `model_cfg`, or None.

    `model_cfg` is the HF config dict (text_config unwrapped). None means "not a model we
    ship numbers for": the recipe still runs, with its own knobs only, and the benchmark
    comparison withholds ratios.
    """
    if not model_cfg:
        return None
    key = (str(model_cfg.get("model_type", "")), int(model_cfg.get("hidden_size", 0)))
    for t in TARGETS.values():
        if t.key == key:
            return t
    return None


def reference(recipe: Recipe | str, model_cfg: dict | None) -> dict | None:
    """What `recipe` measured on the model `model_cfg` describes, or None."""
    r = get(recipe) if isinstance(recipe, str) else recipe
    t = target_for(model_cfg)
    return None if t is None else t.reference.get(r.name)


def describe(name: str | None = None) -> str:
    rs = [get(name)] if name else list(RECIPES.values())
    out = []
    for r in rs:
        out.append(f"{r.name}  [{r.layout}, {r.bpw:.4f} bpw]\n  {r.summary}")
        for t in TARGETS.values():
            ref = t.reference.get(r.name)
            if ref:
                out.append(f"  measured on {t.name}: {ref['decode_tok_s']} tok/s "
                           f"({ref['x_q4km']}x Q4_K_M), {ref['artifact_gb']} GB, "
                           f"medical avg {ref['med_avg']}")
        if r.kernel_env:
            out.append("  kernel env: " + " ".join(f"{k}={v}" for k, v in r.env().items()))
        if r.notes:
            out.append("  " + r.notes.replace("\n", "\n  "))
        out.append("")
    return "\n".join(out)


def describe_targets() -> str:
    out = []
    for t in TARGETS.values():
        out.append(f"{t.name}  [model_type={t.model_type}, hidden={t.hidden}, "
                   f"{t.layers} layers]\n  {t.summary}")
        if t.kernel_env:
            out.append("  kernel tuning: " + " ".join(f"{k}={v}" for k, v in t.env().items()))
        b = t.baseline
        out.append(f"  llama.cpp Q4_K_M on the same slice: {b['decode_tok_s']} tok/s, "
                   f"{b['artifact_gb']} GB, medical avg {b['med_avg']}")
        out.append("  arms with reference numbers: " + ", ".join(t.reference))
        out.append("  measured on:\n    " + t.reference_note)
        out.append("")
    return "\n".join(out)


# ============================================================================ activate
@contextlib.contextmanager
def activate(recipe: Recipe | str, model_cfg: dict | None = None):
    """Run a block with `recipe`'s kernel configuration and NO diagnostic knob set.

    `model_cfg` (the HF config dict) adds the matching target's tuning from TARGETS on top
    of the recipe's pinned knobs; without it, or for a model we do not ship, the recipe
    runs exactly as pinned.  Restores the previous environment on exit, so several arms
    can be exercised in one process (each still pays its own JIT build -- the knobs are
    compile-time macros).
    """
    r = get(recipe) if isinstance(recipe, str) else recipe
    t = target_for(model_cfg)
    saved = {k: os.environ.get(k) for k in (*DIAGNOSTIC_KNOBS, *ARM_KNOBS)}
    try:
        for k in (*DIAGNOSTIC_KNOBS, *ARM_KNOBS):
            os.environ.pop(k, None)
        os.environ.update(r.env())
        if t is not None:
            os.environ.update(t.env())
        yield r
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
