# The kernels

How the CUDA side is put together, which source does what, and what to look at when
porting or tuning. Everything lives in `src/cobaltkernel/csrc/` and is JIT-compiled by
`torch.utils.cpp_extension` — there is no separate build system.

---

## 1. The shape of the design

**One cooperative kernel launch does a whole forward step.** Not one per layer, not one
per matmul. A decode step is exactly one `cudaLaunchCooperativeKernel` with zero
host-to-device copies — the token ids and positions travel *in the kernel parameter
block* (`Args::tok_v` / `pos_v`). A 512-token prefill is one more launch.

The grid is persistent: 256 threads per block, 2 blocks per SM (188 blocks on a 94-SM
slice), and every block walks the whole model. Phases are separated by `cg::this_grid()`
`.sync()`, ten to eleven of them per decoder layer:

```
h → prep_qkv → qkv(fused) → attn (QK-norm + RoPE + KV-append folded in)
  → attn_reduce → o_proj → h2 → prep_gateup → gateup+GeGLU → down_proj
```

then the final norm, `lm_head` and an in-kernel argmax.

**Why a barrier is cheaper than a launch.** Measured on this hardware: a `grid.sync()` at
94–188 blocks costs **0.71 µs**, a kernel launch costs **1.4–1.6 µs**. At a 40 tok/s
target the whole per-token barrier budget is 62 layers × ~6 barriers × 0.73 µs ≈ 0.27 ms,
about 1.1% of the 25 ms token budget. The margin over separate launches is real but it is
~0.7 µs per barrier, not 10×. Keep the grid small: at 564 blocks the same barrier costs
2.2× more.

**Decode is bandwidth-bound and latency-bound at once.** A naive 4-bit dequant GEMV
already reaches 91% of measured memory bandwidth, so there is almost nothing to win from a
cleverer inner loop — the ALU cost of dequantization is about 6%. The win has to come from
*moving fewer bytes*. That is the entire argument for the 16:32 format. At the same time,
each decode phase runs only 0.9–3 waves, so the kernel is latency-bound rather than
throughput-bound, which is why standalone GEMV benchmarks mispredict it (see
[REPRODUCTION.md](REPRODUCTION.md) §7).

---

## 2. Source map

| file | role |
|---|---|
| `cobalt_format.h` | `Mat`, layout ids, `make_mat`, `sub_rows`, `qkv_cs_row`. The format, device-side |
| `cobalt_gemv.cuh` | every dequantizing GEMV. `gemv_blk1632<M, BITS, BPG>` is the 16:32 decoder |
| `gemv_api.cuh` | the small device API the megakernel calls: `gemv_view`, `mat_elem`, `gemv_plain_bf16` |
| `prmt_lut.inc` | the byte-permute selector table used to expand 16 survivors into 32 dense slots |
| `megakernel.cuh` | `MatDesc`, `LayerW`, `Args` — the structs shared with the Python loader |
| `megakernel.cu` | the decode megakernel |
| `bindings.cpp` | its pybind entry point |
| `cobalt_gemm.cuh` | `gemm_tile`, the `mma.sync` bf16 tile path used by prefill |
| `prefill_kernel.cu`, `.cuh`, `prefill_bindings.cpp` | the prefill megakernel |
| `test_gemv.cu`, `test_gemm.cu` | standalone correctness and bandwidth harnesses |

Python side: `runner.py` (decode), `prefill_runner.py` (prefill), `ref_gemma3.py` (the
pure-torch executable specification the kernels must reproduce), `dequant_ref.py` (the
torch oracle that reads the same packed bytes).

---

## 3. The 16:32 decoder

Per 32-column block, one lane:

1. loads the block's `uint32` mask word and its 8 nibble bytes (plus 4 hi2 bytes at b=6)
   — all at **fixed offsets**, because the survivor count is a constant;
2. computes four independent funnel shifts from the mask word
   (`o_i = popcount(mask & ((1 << 8i) − 1))`) instead of a 4-deep loop-carried chain;
3. expands the 16 survivor codes into 32 dense byte slots with two `__byte_perm` (`prmt`)
   operations against a selector looked up from `prmt_lut.inc`;
4. accumulates `scale · (Σ_kept e_p·x_p − zero · Σ_kept x_p)` over the block.

Step 4 is why pruned positions contribute exactly zero regardless of the zero point
([FORMAT.md](FORMAT.md) §5).

Four optimisations that mattered, each measured on the real kernel:

| change | effect |
|---|---|
| selector LUT staged in `__shared__` once per block | **+11.9%** |
| zero-fill of pruned slots cancelled against the accumulator | **+12.7%** |
| `__byte_perm` straight into a float mantissa, deleting the `cvt` | +1.9% |
| KV appended once per key-split rather than per block | +0.2% |

And one that did not: putting the selector LUT in `__constant__` memory was a **large
negative** — the indices are divergent, and constant-cache serialises on divergent access.
It was the obvious first thing to try.

---

## 4. Tuning knobs

`src/prod/recipes.py` pins the shipped values; these are the ones worth revisiting on new
hardware. Everything is read from the environment at build time (they are compile-time
macros) or at runner construction.

| knob | shipped | what it does |
|---|---|---|
| `COBALT_BLK1632` | 0 / 4 / 6 | which block layout the kernel is compiled for |
| `COBALT_BLK1632_O` | 0 for hybrids | the prefill `gemm_tile`'s layout for `o_proj` (compile-time, unlike decode, which dispatches per matrix at runtime) |
| `COBALT_BLK1632_LUT` | 2 | selector table placement: 0 global, 1 `__constant__`, 2 `__shared__`, 3 arithmetic |
| `COBALT_KPB` | 128, **256 for DENSE4** | keys per attention block. Sets the cross-block key split |
| `COBALT_MINB` | 2 | `__launch_bounds__` min blocks per SM, i.e. the per-thread register budget |
| `COBALT_BLOCKS` | 2 per SM | grid size |
| `COBALT_GATEUP_RP` | auto | rows per warp in the fused gate/up phase |

> **Do not export a knob whose name contains `NOOVH`, `HALFJ`, `NOLD`, `NOXB`, `XSYNC` or
> `ATTN_DIAG`.** Those are diagnostics that delete real work to price it, and they produce
> wrong numerics silently. `prod` clears all of them before every run.

Some measured guidance on the tuning axes, so you do not re-derive it:

* **`COBALT_KPB` is exhausted for the block arms** — 47.22 at 128 versus 47.20 and 47.19
  at 160 and 256. The "idle warps" hypothesis is false: fewer warps per block trades
  exactly against more blocks. KPB=64 is *worse*, because the cross-block reduce grows
  faster than the walk shrinks. DENSE4 alone prefers 256, by 0.55%.
* **Barriers are not the bottleneck.** An `XSYNC` slope prices the entire per-layer
  barrier cost at ~0.48 ms per token, not the ~1.9 ms an earlier estimate assumed.
* **The load-instruction axis is worth +0.1–0.9%**, not the +22% an early diagnostic
  suggested. That diagnostic deleted 4 of 12 bytes along with the load, so it priced
  *bytes*, not instructions. Interleaving the planes to cut loads inverts to −4.6% at b=4.

---

## 5. Porting

**To another GPU.** Nothing is hard-coded to Blackwell. `prod.env.configure()` detects the
compute capability and emits the right arch flag, including the `a` suffix on the
arch-specific targets (9.0, 10.0, 12.0). Requirements are compute capability 8.0+ (for
`mma.sync.m16n8k16.bf16` in the prefill path) and cooperative-launch support. Expect
decode tok/s to track memory bandwidth: the kernel reaches 91–98% of measured read
bandwidth depending on layout.

Note that this is a **warp-level `mma.sync` architecture** design — Ampere/Ada style, not
Hopper's warpgroup model. `wgmma.mma_async` is not used and does not compile for sm_120.
On Hopper you would leave the decode path alone (it is bandwidth-bound and would gain
nothing) and could rewrite `gemm_tile` around `wgmma` for prefill.

**To another model.** The Gemma3-specific parts are: the RoPE tables (two frequency sets,
local and global), the sliding-window attention pattern, the QK-norm, the pre/post
feed-forward norm pair, the embedding scale, and the tied embedding / lm_head.
`ref_gemma3.py` is the executable specification of all of it in ~300 lines of readable
torch — start there, get the reference right first, and only then change the kernel. The
GEMV, the format and the packer are architecture-independent.

**Verification discipline.** Whatever you change, run
`python -m prod verify <artifact> --config <hf> --kernel` and hold the 98.5% argmax bar
against a torch oracle built from the *same packed bytes*. Batch M=4 must stay bit-exact
against four independent M=1 runs — if it is not, some part of the reduction order has
started depending on the batch size, which is a correctness bug even when the logits look
close.
