# CBK1 — the CoBALT packed weight format

A CBK1 directory is a **byte-for-byte image of what the kernel reads**. Loading a model
is `open()` + `cudaMemcpy` with no unpacking and no host-side reshuffle, so disk bytes
equal VRAM bytes and the number in `manifest.json` is the number `nvidia-smi` will show.

This document is complete enough to write your own reader. The reference reader in torch
is `src/cobaltkernel/dequant_ref.py` (~150 lines) and the device-side one is
`src/cobaltkernel/csrc/cobalt_gemv.cuh`.

---

## 1. Container

```
manifest.json      format, version, config, and every array's offset and length
layer_00.bin …     one blob per decoder layer: the 6 quantized matrices
embed.bin          the tied embedding / lm_head
misc.bin           every RMSNorm weight, fp32
```

Each `.bin` is a concatenation of **256-byte-aligned arrays**. Every matrix's `data` array
carries **64 bytes of tail padding**, so a decoder may over-read a few words past the last
row.

`manifest.json` records, per matrix:

```json
{"K": 4096, "N": 5376, "G": 42, "layout": 6, "layout_name": "BLK1632_4",
 "cs_rows": 1, "n_survivors": 11010048, "stream_bytes": 8355840, "bpw": 3.1875,
 "arrays": {"data":  {"off": 0,       "bytes": 8257536},
            "scale": {"off": 8257792, "bytes": 344064},
            "zero":  {"off": 8602112, "bytes": 172032},
            "col_scale": {"off": 8774400, "bytes": 10752}}}
```

`K` is the output-row count, `N` the input-column count. Everything is row-major.

---

## 2. Dequantization

```
W_hat[k, j] = (q[k, j] − zero[k, j/128]) · scale[k, j/128] · col_scale[j] · keep[k, j]
```

* **`scale`** fp16, **`zero`** uint8, one pair per `(row, group of 128 input columns)`,
  stored in two separate row-major `[K, G]` arrays — not interleaved with the codes, so
  they form their own perfectly coalesced stream.
* **`col_scale`** fp16 `[N]`, one per input column. The kernel never applies it to a
  weight; it folds it into the **activation** (`x' = c ∘ x`) once per matrix, which is
  free — 10.5 KB for N=5376, read once and L2-resident.
* **`keep`** is the survivor mask: 1 for a kept weight, 0 for a pruned one.

> ⚠ **Every matrix has its own `col_scale`.** q, k and v read the same hidden state but
> have three different `c` vectors, and so do gate and up. A kernel that fuses them must
> stage one column-scaled copy of `x` per constituent matrix. This is intrinsic to
> CoBALT's per-matrix column normalisation and cannot be folded into the group scale,
> because `c` varies *within* a group.

---

## 3. Layouts

| id | name | bpw | row stride | codes |
|---|---|---|---|---|
| 1 | `DENSE4` | 4.1875 | `N/2` | a 4-bit code for every position; pruned ones hold `zero` |
| 2 | `DENSE8` | 8.1875 | `N` | 8-bit, used for the embedding quality arm |
| 4 | `BF16` | 16 | `2N` | passthrough, for ablations |
| 6 | `BLK1632_4` | **3.1875** | `3N/8` | 16:32 block mask, 4-bit survivors |
| 7 | `BLK1632_6` | **4.1875** | `N/2` | 16:32 block mask, 6-bit survivors |

Ids 0, 3 and 5 are the variable-length bitmap-sparse layouts (`SPARSE4`, `SPARSE4X`,
`SPARSE4E`). They exist in the tree and are **not shipped**: see §6.

All shipped layouts have a **fixed row stride** and no per-row offset table.

bits per weight = codes + mask + `24/128` for the fp16 scale and uint8 zero of each
128-group. So `BLK1632_4` = 2.000 + 1.000 + 0.1875, and `BLK1632_6` = 3.000 + 1.000 +
0.1875 = **exactly DENSE4's 4.1875** — the matched-memory arm really is matched, to the
byte.

### 3.1 DENSE4

Nibble `j` of a row lives in byte `j >> 1`, low nibble for even `j`. Pruned positions
store the group's `zero`, so they dequantize to exactly 0.0 — *provided the packer put the
grid's zero on a representable code*. See §5.

### 3.2 BLK1632 — three contiguous planes per row

`NB = N/32` blocks per row. Every aligned 32-column block holds **exactly 16 survivors**,
which is what makes every offset a compile-time constant: block `t`'s codes are always at
byte `8t` of the nibble plane. No prefix popcount, no offset table, no sliding shift chain.

```
row r at data + r * stride            stride = 3N/8 (b=4), N/2 (b=6)

  [0,     N/8)   MASK plane    one uint32 per block, LSB-first; popcount == 16 always
  [N/8,  3N/8)   NIBBLE plane  8 B per block = the LOW 4 bits of the block's 16 survivor
                               codes in increasing column order; survivor 2j is the low
                               nibble of byte j
  [3N/8,  N/2)   HI2 plane     b=6 only. One uint32 per block:
                                 bits [0,16)  high-2-bits of survivors at EVEN
                                              within-block index 0,2,…,14 (slot e at
                                              bits [2e, 2e+2))
                                 bits [16,32) the same for ODD indices 1,3,…,15
```

Two design choices worth understanding if you write a reader:

* **Planes, not a per-block interleave.** At b=6 an interleave is a clean 16 B per block,
  but at b=4 it is 12 B — so a 64-bit code load would land on a 4-byte boundary, which is
  illegal. Separate planes make every stream individually contiguous and naturally
  aligned, and let a decoder choose its own per-lane granularity (1, 2 or 4 blocks)
  without the format changing.
* **The hi2 parity split.** The byte-permute pool the decoder uses wants
  `{c_o, c_{o+2}, c_{o+4}, c_{o+6}}` in one operand and the odd ones in the other, at a
  *variable* offset `o`. Splitting the 2-bit codes by within-block index parity makes each
  of those a contiguous 8-bit field of one 16-bit half — two shifts, no de-interleave.

`N % 128 == 0` for every shape in these models, so `N/8`, `3N/8` and `N/2` are all
multiples of 16: every plane and every row starts 16-byte aligned with zero padding.

---

## 4. Fused matrices

The packer emits two row-fused matrices per layer. Fusion changes nothing about the byte
layout — a fused blob is simply the same layout applied to a taller matrix — but it lets
one kernel phase stream one large matrix instead of several small ones, which matters:
the small `k_proj` and `v_proj` do not fill the tile grid on their own (100–295 GB/s
separately versus **431 GB/s** fused).

**`gateup`** — `K = 2 · intermediate`, rows **interleaved**: row `2i` is gate row `i`, row
`2i+1` is up row `i`, so a warp computing both can apply GeGLU immediately.
`col_scale` is fp16 `[2, N]`.

**`qkv`** — rows **concatenated**: q rows, then k rows, then v rows. `col_scale` is fp16
`[3, N]`, and the kernel picks the row from the output row index.

Because the row stride is fixed, each projection's rows are a *contiguous* byte range of
the fused arrays. The packer therefore writes `data`, `scale` and `zero` **once**, and the
`q_proj` / `k_proj` / `v_proj` manifest entries carry `alias_of: "qkv"` with offsets
pointing *into* the fused arrays. Only each projection's own `col_scale` is duplicated.

**Consequences for a reader:**

* Any offset-driven loader reads the aliases correctly with no special case.
* When you total bytes or parameters, **skip entries that have `alias_of`**, or you will
  count the fused matrix twice.
* `cs_rows` tells you the `col_scale` shape: 3 for `qkv`, 2 for `gateup`, 1 otherwise.

---

## 5. The zero-point trap

An asymmetric quantizer stores a zero-point, so a pruned position decodes as
`(0 − zero) · scale`, which is **not zero** unless the grid's zero is representable. The
packer re-derives a 0-inclusive hull for exactly this reason.

Measured behaviour, with a deliberately off-grid zero point and a huge activation at every
pruned column:

| layout | pruned column contributes | with the zero point forced off-grid |
|---|---|---|
| `BLK1632_4` / `BLK1632_6` | exactly 0.0 | still exactly 0.0 |
| `DENSE4` | 4.07e−02 of fp32 residue | **9.30e+06** |

The block layouts store *no code at all* for a pruned position, and the decoder applies
the zero point only over kept slots — so a pruned column contributes exactly zero
regardless of what `zero` holds. DENSE4's cancellation happens in the final accumulator
subtraction instead, which leaves fp32 residue and blows up entirely if the hull was not
re-derived. Neither is a problem in the served model (activations are O(1)), but it is the
honest statement of what "bit-exact zeros" means for each layout, and it is why
`python -m prod verify` checks pruned positions rather than trusting them.

---

## 6. Layouts that exist and are not shipped

`SPARSE4`, `SPARSE4X` and `SPARSE4E` store survivor codes only, with a bitmap and either a
warp prefix-scan or an explicit per-group offset table. They are 3.19–3.33 bpw — fewer
bytes than DENSE4 — and they are **slower**, reaching only 45–66% of memory bandwidth
against DENSE4's 95–98%, because the code stream is variable-length: the decoder needs a
per-lane prefix popcount *and* a loop-carried 64-bit sliding-shift chain.

That negative is the reason `BLK1632` exists. Constraining the mask to exactly 16
survivors per 32 columns costs a little quality (0.28 pt of task average at 4 bits) and
converts a variable-length format into a fixed-stride one — which is what finally made the
sparsity payable in bytes *and* in wall-clock.

The unshipped layouts remain in the tree because they are the control that establishes the
result. Do not deploy them.

---

## 7. Sizes, MedGemma-27B

62 layers; q 4096×5376, k/v 2048×5376, o 5376×4096, gateup 43008×5376, down 5376×21504;
25,598,361,600 decoder parameters; embedding 262144×5376 packed DENSE4.

| | DENSE4 | BLK1632_4 | BLK1632_6 |
|---|---|---|---|
| decoder stack | 13.399 GB | **10.199 GB** | 13.399 GB |
| total artifact | 14.145 GB | **10.945 GB** | 14.145 GB |
| bpw | 4.1875 | **3.1875** | 4.1875 |

Reference: llama.cpp Q4_K_M is 16.55 GB at 4.81 bpw.

## 7b. Sizes, BioMistral-7B

32 layers; q 4096×4096, k/v 1024×4096, o 4096×4096, gateup 28672×4096, down 4096×14336;
6,979,321,856 decoder parameters; embedding 32000×4096 and the **untied** `lm_head`
(its own `lm_head.bin`, DENSE4) packed separately; 2 RMSNorms per layer in `misc.bin`.

| | DENSE4 | BLK1632_4 |
|---|---|---|
| decoder stack (streamed per token) | 3.653 GB | **2.781 GB** |
| total artifact | 3.79 GB | **2.92 GB** |
| bpw | 4.1875 | **3.1875** |

Reference: llama.cpp Q4_K_M is 4.37 GB at 4.8 bpw. The same `blk1632_b4` recipe and the
same CBK1 layout produce both models' artifacts; only the shapes, the untied head and the
norm set differ, all of which the manifest records.

The embedding is quantized with plain asymmetric group-128 min-max RTN — no mask, no
compensation, no column scale — because it is read as a lookup, not multiplied. Its
relative error is 0.1036 at 4 bits and 0.0061 at 8 bits; the 8-bit option costs +0.70 GB
of artifact *and* +0.70 GB streamed per token, since the tied lm_head is read in full at
every decode step.
