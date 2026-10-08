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

**To another model.** Two families are wired (§6): Gemma3 and the llama family
(Mistral / Llama). The family-specific parts are the RoPE tables, the sliding-window
pattern, the QK-norm, the norm set per layer, the activation, the embedding scale and
whether the lm_head is tied. `ref_gemma3.py` and `ref_llama.py` are the executable
specifications, each ~300 lines of readable torch — start there, get the reference right
first, and only then change the kernel. The GEMV, the format and the packer are
architecture-independent; `arch.py` is where a new family's flags go.

**Verification discipline.** Whatever you change, run
`python -m prod verify <artifact> --kernel` and hold the 98.5% argmax bar against a torch
oracle built from the *same packed bytes*. Batch M=4 must stay bit-exact against four
independent M=1 runs on the row-loop build — if it is not, some part of the reduction
order has started depending on the batch size, which is a correctness bug even when the
logits look close. The one sanctioned exception is the tensor-core decode build
(`COBALT_BLK1632_MMA=1`, BioMistral-7B): M=1 decodes qkv/o_proj on `mma` with f32 slice
sums while M>1 keeps the row loop, so the gate there is identical greedy tokens plus a
0.5 bound on max |logit diff| (measured 0.09–0.14, one bf16 ulp at these magnitudes).
An exact tie in the reference's top-2 logits counts as agreement in every gate: bf16
logits near 16–32 sit on a 0.125 grid and the reference cannot adjudicate a tie; the
free-running greedy gate may fork at one (prefix must match), and the teacher-forced
gate is what asserts every decode step.

---

## 6. Model families and the in-kernel generation loop (2026-10-06)

The kernels now serve two families, read from `config.json` (`src/cobaltkernel/arch.py`):

| | gemma3 (medgemma-27b, gemma-3-4b) | llama (BioMistral-7B / Mistral-7B, Llama) |
|---|---|---|
| norms per layer | 4, `bf16(x·rr·(1+w))` | 2, `bf16(bf16(x·rr)·w)`; `post_attn_ln`/`post_ff_ln` null → plain residual add |
| QK-norm | yes | bypassed (null pointer) |
| RoPE / sliding | local+global θ, 1-in-N sliding | one θ, uniform sliding window (Mistral) or none |
| activation | GeGLU (`act_gelu=1`) | SwiGLU (`EPI_SWIGLU_BF16`) |
| lm_head | tied to the embedding | separate `lm_head.bin` (DENSE4/DENSE8) |
| query:kv ratio | KG=2 | KG=4 — a **compile-time** `CBK_KG` picked from the config; a mismatch is a hard error |

All of this is runtime-switched except `CBK_KG`; the Gemma3 build is bit-identical to the pre-port kernel
(`torch.equal` over 48 decode steps, `scripts/run_port_bitident.sh`). The executable spec for the llama family is
`ref_llama.py` (fp32: `max|dlogit| = 0` vs transformers).

**In-kernel greedy generation.** `MegaRunner.generate` / `KernelRunner.generate_inkernel(tokens, positions, n)` emits
`n` tokens from ONE cooperative launch: the kernel feeds its own in-kernel argmax, advances positions and recomputes
the attention key-split on device (per-step state lives in a `__shared__ Step`; never write to the by-value `Args`,
it spills the whole parameter block). Tokens are identical to the one-launch-per-token path (127/127 in every
record) and the decode rate becomes immune to host load — on a shared host that was worth 0–15%.

**Knobs measured on BioMistral-7B (2g slice, GPU-side `clock64` step).** Ship:
`COBALT_BLOCKS=160`, `COBALT_KPB=128`, `COBALT_ATTN_KUNROLL=4`, `COBALT_PVB=2` (the KG=4 build is at the 128-register
bound; these un-spill it), stock R set. Measured NO on this part: KPB 32–64/256/1024, PVB 8/16, `COBALT_RMAX` 1/2,
gate|up RP=1/split/RP=4, `COBALT_BLK1632_PFX=2`, MSCHED, MINB=1, o_proj prefetch, `COBALT_ATTN_COMBINE1`,
`COBALT_CSPLIT=2` (wins on a 1g slice only), `COBALT_FUSE_RESID` 1/2, and the extended one-wave R set
(`COBALT_RMAX=8`: R∈{3,5,6,8} for o/down — the fused qkv phase must keep a power-of-two R). Result of record:
llama.cpp Q4_K_M 143.7 tok/s vs this kernel 138.4 in-kernel (0.963×) at 1.50× fewer bytes; +12.4% over the
kernel's own DENSE4 layout.

**Tensor-core decode of the L2-resident phases (2026-10-07, `csrc/cobalt_gemv_mma.cuh`).** The 16:32 planes are
unchanged; `COBALT_BLK1632_MMA=1` runs the selected GEMV phases on `mma.m16n8k16` (bf16 × bf16 → f32): the shipped
expansion produces the 8 code bytes per bitmap byte, one biased `PRMT` turns each pair into a bf16×2 A-fragment
word (bf16 `0x43cc` is exactly `128 + cc`, so the bias of `COBALT_BLK1632_BIAS` carries over), x′ is loaded as bf16
pairs straight into B (gate in column 0, up in column 1 — the interleaved gate|up rows need no repack), Σx′ per
group comes from a second mma with an all-ones A, and the per-group `s·(C − (128+z)·Σx′)` epilogue runs on the C
fragment. A warp owns a 16-row tile × column slice; slices of one tile are summed with f32 atomics and the
last-arriving warp runs the epilogue (`Args.mpart` / `Args.mcnt`, kept zero between phases). Static cost: 155 SASS
per 32-column row-granule vs 230 for the row loop. Measured: it **loses** on the DRAM-streaming matrices (gate|up,
down, lm_head — its compute alone is slower than the row loop with its loads: latency-bound dependency chains, and
every warp-level load touches 8 rows) and **wins** on the two phases that run L2-resident (qkv, o_proj), hence the
per-phase mask `COBALT_MMA_PHASES` (bits: 1 qkv, 2 o_proj, 4 gate|up, 8 down, 16 lm_head; default 3). Prefetch
depth `COBALT_MMA_PF_L2` (1 for those phases) / `COBALT_MMA_PF` (4 for streaming ones); unit sizing
`COBALT_MMA_UPW` (target work units per warp, 2 — the per-unit fence + counter dominated o_proj at 4) with
`COBALT_MMA_UPW_O=1` for o_proj alone (its K is the hidden size, the fewest tiles: it wants the largest units). Measured neutral or worse: `COBALT_MMA_ACC2` (two accumulator chains),
`COBALT_MMA_COAL` (coalesced smem-staged loads: fixes the load pattern but its 30 KB/block buffer evicts attention's
L1), lm_head on the MMA path. On BioMistral-7B this took the 2g-slice step from 7.20 to 6.80 ms with the oracle
verify unchanged (99.0–99.3 % argmax; the fp32 slice sums make tokens order-nondeterministic, so the in-kernel/
step-path token match is no longer a correctness signal — the verify is). The prod facade ships these knobs for the
mistral family (`recipes.MODEL_TUNING`).

**Round 3 (2026-10-08): decode attention on tensor cores, a cheaper expansion table, and a lookahead in the row decoder.**
Same artifact and bytes; on BioMistral-7B the 2g-slice step went 6.84 → 6.15 ms (146 → 164 tok/s in-kernel, 1.14× llama.cpp
Q4_K_M in the same window), all three verified with the record oracle protocol (tie-free prompt seed: every gate PASS, M=4
bit-exact). Shipped for the mistral family in `recipes.MODEL_TUNING`:

* `COBALT_ATTN_TC=1` — `phase_attn_tc`: S = Q·Kᵀ on `mma.m16n8k16` (A = the KG query heads of the GQA group padded to 16
  rows, B = 8 keys straight from the K cache; the mma's k index is a permutation of the dims so each lane loads 8 contiguous
  bytes per k-step), the online softmax on the C fragment (lane (g,t) owns 4 keys of head g), then Oᵀ = Vᵀ·Pᵀ with B = Pᵀ taken
  from S's C fragment unchanged and A = Vᵀ via `movmatrix` transposes of 4-byte V loads. 16 keys per warp-chunk, every warp
  of every block busy. The 8 warp states merge in one round through fp16 slots (`CBK_ATTN_TC_HALF`), and the last split
  block of a kv-head does the cross-split reduce (`CBK_ATTN_TC_REDUCE`), so the `attn_reduce` phase and its barrier are gone.
  attention 1.15 → 0.57 ms per token. **Shared memory is the constraint**: on GB202 L1 and smem share 128 KB per SM, and two
  blocks × (static + dynamic) crossing the 32 KB carve-out step costs every GEMV phase 3–10 % (measured by inflating the
  shipped build's dynamic smem alone) — hence the fp16 slots (8 KB) rather than f32 (16 KB); a 3-level register tree
  (`CBK_ATTN_TC_TREE`) halves the smem too but its six `__syncthreads` cost more than it saves. The in-fragment q-RoPE
  (`CBK_ATTN_TC_QROPE`) and hoisting all V loads (`CBK_ATTN_TC_VPRE`) are measured negatives (registers).
* `COBALT_BLK1632_LUT=4` — the per-bitmap-byte table holds `{sel0, sel1, mk0, mk1}` as one `uint4`, so the pruned-slot byte
  masks come from the table instead of two bit-spread multiplies per byte: −6 of ~22 ALU ops per 8-position expansion, and
  the ALU pipe (16 lanes per partition, ~70 % of the decoder's SASS) is the decoder's binding pipe. −2.5 % step.
* `COBALT_BLK1632_MSCHED=3 COBALT_BLK1632_PFH=2` — masks and nibbles of batch k+1 are issued before batch k is decoded (the
  lookahead that makes a warp's own DRAM latency overlap its issue-bound decode), at PF 1 so the second buffer costs the
  shipped register footprint (at PF 2 it spills: 7.6 ms). −3.2 % step. A cp.async shared-memory ring that does the same
  decoupling (`csrc/cobalt_gemv_stream.cuh`, `COBALT_STREAM`) is a large negative here — smem carve-out plus issue slots.

Measured negatives kept in-tree, default off: contiguous equal-work chunking of the walks (`COBALT_CHUNK`; the split-group
atomics land at the end of every warp's work), the MMA tile loop on the streaming phases even with its scale/zero
prefetched (`COBALT_MMA_SZPRE`, three configurations), and the items above. Measurement rules that this round
re-learned: interleave shipped / control / candidate in ONE sweep (a build's phase times are not comparable across windows),
and judge correctness by the oracle verify — the in-kernel-vs-step token match is a near-tie coin flip under the MMA slice
sums (the shipped build flips it too).
