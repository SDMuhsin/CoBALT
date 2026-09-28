# Reproduction guide

Everything needed to build a CoBALT artifact from a Hugging Face checkpoint, verify it,
and serve it with the custom CUDA kernels on your own GPU server.

Read the whole of §1 before starting: §1.3 in particular will save you an afternoon.

---

## 0. If you were sent the big-files overlay, skip stages 1 and 2

The repository does not track two things you need: the calibration set (`.gitignore`
excludes `results/`) and the compressed weights themselves. We ship those separately as a
**drop-in overlay** — unpack it at the root of your clone and it lands exactly where the
code already looks for it. It contains no code and no documentation, so there is nothing
to reconcile against the clone.

```bash
git clone <repo> && cd <repo>
tar -xf cobalt-bigfiles.tar        # AT THE REPO ROOT; adds results/ and artifacts/
sha256sum -c SHA256SUMS            # optional; 11 GB transfers do get truncated

export PYTHONPATH=$PWD/src
python -m prod doctor                                       # §1.3
python -m prod verify   artifacts/medgemma-27b-<arm>        # §5, ~2 s, no GPU
python -m prod generate artifacts/medgemma-27b-<arm> --prompt "A 54-year-old presents with" -n 64
python -m prod bench    artifacts/medgemma-27b-<arm> --prompt 512 --gen 128 --out bench.json   # §7
```

What the overlay adds:

| path | what it is | needed for |
|---|---|---|
| `artifacts/medgemma-27b-<arm>/` | packed CBK1 weights, plus `config.json` and the tokenizer | everything below |
| `results/accel4bit/calib_*` | the calibration set the recipes pin | **only** if you re-run stage 1 |

**The weights are already built, so stages 1 (§3) and 2 (§4) are already done** — nothing
on your side runs the quantizer, and reproducing our numbers does not touch the
calibration file. `--config` is not needed either: `config.json` and the tokenizer sit
beside the weights, so no Hugging Face download is required. The recipe is read out of
`manifest.json`, so an artifact cannot be paired with the wrong kernel configuration.

Read §1 (prerequisites), §5 (what verify proves), §7 (the measurement protocol) and §8
(where this design does not win). §3 and §4 are background — read them to understand what
produced the weights, not as steps to perform.

This is the path we recommend for reproducing our published numbers: it takes both the
calibration data and the quantizer out of the set of things that could differ between us,
so a disagreement can only come from the kernel or the hardware.

---

## 1. Prerequisites

### 1.1 Hardware

| requirement | why |
|---|---|
| NVIDIA GPU, compute capability **8.0+** (Ampere or newer) | the prefill kernel uses `mma.sync.m16n8k16.bf16`, which is sm_80+ |
| **cooperative launch** support | the whole design is one `cudaLaunchCooperativeKernel` per step |
| VRAM ≥ artifact + KV cache | MedGemma-27B: 11.0–14.2 GB of weights, plus ~0.48 MiB per token of context |

Our numbers were taken on one **MIG 2g.48gb slice** of an RTX PRO 6000 Blackwell
(compute capability 12.0, 94 SMs, 770.8 GB/s measured read bandwidth). Any Ampere,
Hopper or Blackwell card will run the kernels; the achieved tok/s scales with memory
bandwidth, since decode is bandwidth-bound.

**A 24 GB card is enough for MedGemma-27B**, with room to spare. The KV term is 0.484 MiB
per token and is identical for every arm, so for the 11 GB arms:

| context | weights | KV cache | total |
|---|---|---|---|
| 2 048 | 10.4 GB | 1.0 GB | 11.4 GB |
| 4 096 | 10.4 GB | 1.9 GB | 12.3 GB |
| 16 384 | 10.4 GB | 7.7 GB | 18.1 GB |

Note that **serving needs far less memory than quantizing**. §3 streams a 27B checkpoint
through a 48 GB slice; if your card is smaller than that, take the prebuilt-package route
in §0 rather than trying to build the artifact locally.

### 1.2 Software

```
python >= 3.10
torch  >= 2.4    (built against the same CUDA major version as your nvcc)
CUDA toolkit     nvcc on PATH, or CUDA_HOME exported
ninja            pip install ninja   -- torch shells out to the BINARY
safetensors, numpy
transformers     only for quantization and the tokenizer helpers
```

The CUDA extension is JIT-compiled by `torch.utils.cpp_extension` on first use and cached
afterwards in `TORCH_EXTENSIONS_DIR`. There is nothing to `pip install` and no build step
of your own. **Budget 10–20 minutes for that first build**, not seconds: `megakernel.cu`
is one large, heavily-templated translation unit and nvcc is single-threaded on it. It is
compiling even though nothing is printed — see the stale-lock warning in §9 for how to
tell a real compile from a blocked one. Each distinct recipe, and each batch size, gets
its own build directory and so its own first build.

### 1.3 Check the machine before anything else

```bash
export PYTHONPATH=/path/to/repo/src
python -m prod doctor
```

This prints the detected compute capability, SM count, nvcc release, ninja location and
the build environment it will use, and exits non-zero with an actionable message for each
blocker it finds. Fix everything it reports before continuing — every failure mode it
checks for produces a confusing error much later otherwise.

Two that bite people repeatedly:

* **`ninja` not found.** It is installed into your venv's `bin/`, and torch invokes the
  binary by name — so calling `…/venv/bin/python` by absolute path is *not* enough, the
  venv's `bin` has to be on `PATH`.
* **Cache directories on a small home volume.** The JIT build cache, the CUDA cache and
  the HF cache all land under `$HOME` by default. Point `COBALT_CACHE` (and `HF_HOME`,
  `TMPDIR`) at a large volume if your home quota is tight.

**What `doctor` does not check.** It validates the toolchain and the device: compute
capability, nvcc's presence, ninja, cooperative launch. It does *not* check that your
nvcc is new enough to emit code for the compute capability it just detected, nor that
free VRAM exceeds the artifact, nor that any input file exists. So a clean
`"blockers": []` means "the machine looks sane", not "the next command will work" — and
an `Unsupported gpu architecture 'compute_XXa'` from nvcc is a toolkit too old for your
card, which doctor will have reported as fine. Match your CUDA toolkit to your GPU
generation, not merely to torch.

You can also check the package itself, which needs no GPU and no artifact:

```bash
python -m prod.selftest
```

It confirms that every recipe still reconstructs the exact command line that built the
shipped arms, and that the diagnostic knobs really are cleared. Run it after changing
anything under `src/prod/`.

### 1.4 Layout

```
src/prod/            this package -- one entry point per stage
src/cobaltkernel/    the implementation: quantizer, packer, CUDA sources, reference model
  csrc/              the CUDA kernels (see docs/KERNELS.md)
docs/                this documentation
```

`src/prod` deliberately contains **no copy** of the algorithms or the CUDA code. It is a
façade: one supported way to run each stage, one pinned configuration per shipped arm,
and the checks that tell you whether a build is good. Both directories are required.

---

## 2. Pick a recipe

```bash
python -m prod recipes                 # all four, with what each one measured
python -m prod recipes blk1632_b4      # just one
```

A **recipe** is one deployable arm, pinned end to end: quantizer flags, packer flags, the
kernel's compile-time and runtime knobs, and the numbers we measured for it. Nothing in
this package runs without one, and the artifact's own manifest is enough to infer which
recipe built it (`prod.recipe_for_artifact`), so an artifact cannot be paired with the
wrong kernel configuration by accident.

If you have no strong preference, use **`blk1632_b4_oproj`**: the best quality per byte of
the four, within 0.4% of the fastest arm's decode rate.

> **Why recipes are not just documentation.** The kernel has about 25 environment knobs,
> and six of them (`COBALT_BLK1632_NOOVH`, `COBALT_ATTN_DIAG`, `COBALT_BLK1632_HALFJ`,
> `COBALT_BLK1632_NOLD`, `COBALT_XSYNC`, `COBALT_BLK1632_NOXB`) are *diagnostics that
> deliberately compute the wrong answer* — they exist to price a lever by deleting the
> work it does. A stray export of any of them gives you a fast model that quietly
> generates worse text, and nothing complains. Every entry point here runs inside
> `recipes.activate()`, which sets the pinned knobs and **unsets every diagnostic**.
> There is no supported way to reach a diagnostic build through `prod`.

---

## 3. Stage 1 — quantize

```bash
python -m prod quantize \
    --recipe blk1632_b4_oproj \
    --model  /models/medgemma-27b-text-it \
    --out    /artifacts/medgemma-27b-raw
```

Add `--dry-run` first to see the exact underlying command.

The model is never materialised: decoder layers are constructed on `meta` and their
weights streamed from the snapshot one layer at a time, with calibration activations held
in pinned host memory. A 27B model therefore quantizes inside a 48 GB slice — but it takes
**hours**, so run it detached:

```bash
setsid nohup python -m prod quantize ... > quant.log 2>&1 < /dev/null & disown
```

Progress is one line per layer in the log. Output: `layer_XX.safetensors` per decoder
layer plus `manifest.json` recording the full quantizer configuration.

### What the quantizer does

Per matrix, using a calibration set of 128 sequences × 2048 tokens:

1. **Wanda importance** `|W_ij| · ‖X_j‖₂` — magnitude weighted by how much the input
   column actually carries.
2. **Row + column quantile self-normalisation** (the "column balance", exponent β = 0.5).
   This is the part that is CoBALT's own: dividing by per-row and per-column quantiles
   before selection stops any single input column from being starved of survivors.
3. **Selection** — either one global top-k over the matrix (`dense4`), or, with
   `mask_block=32`, exactly **16 survivors in every aligned block of 32 input columns**.
   Both give exactly 50% sparsity; only the constraint differs.
4. **OBS compensation** — the pruned mass is pushed into the surviving weights using
   `H = XᵀX + λI` from the calibration pass, so the layer's *output* is preserved rather
   than its weights.
5. **Per-column scale + group-128 asymmetric RTN** over the survivors.

Steps 1, 2, 4 and 5 are identical across all four recipes. The only thing that changes is
step 3, and — for `blk1632_b6_oproj4` — the code width.

### Calibration data

> **This will stop you on a fresh clone.** The recipes pin
> `results/accel4bit/calib_ultrachat_512x2048.txt`, and `results/` is excluded by
> `.gitignore` — so the file is **not present in the repository**, and stage 1 fails on it
> with `FileNotFoundError` before any work is done. `doctor` does not check for it. Pick
> one of:
>
> * `--calib wikitext2` — built in, downloads itself, needs no file. Easiest, but it is
>   *not* what our published arms were calibrated on, so expect small quality differences.
> * `--calib-file <your own text>` — one sample per blank-line-separated block. **Best
>   choice for deployment:** calibrate on traffic that resembles yours.
> * Regenerate ours with `src/accel4bit_dump_calib.py`, which pulls
>   ultrachat_200k/train_sft and packs it identically (seed 42, 2048 tokens).
>
> Or avoid the question entirely: the big-files overlay (§0) ships this exact file at
> this exact path, and reproducing our numbers never runs this stage anyway.

The recipes point at `results/accel4bit/calib_ultrachat_512x2048.txt` (512 sequences from
ultrachat_200k/train_sft, seed 42, packed to 2048 tokens; the first 128 are used). **Calibrate on text that
resembles your deployment traffic** — the quantizer's compensation step fits the input
covariance, and a mismatch shows up as a quality loss no amount of kernel work recovers.

---

## 4. Stage 2 — pack

```bash
python -m prod pack \
    --recipe blk1632_b4_oproj \
    --raw    /artifacts/medgemma-27b-raw \
    --out    /artifacts/medgemma-27b-cbk1 \
    --model  /models/medgemma-27b-text-it
```

Minutes, not hours. `--model` is required because the tied embedding / lm_head and the
RMSNorm weights come straight from the HF snapshot — the quantizer only writes the seven
decoder linears.

The output directory is a **byte-for-byte image of what the kernel reads**: loading is
`open()` + `cudaMemcpy`, with no unpacking and no host-side reshuffle, so disk bytes ==
VRAM bytes. `manifest.json` gives every array's offset and length. The command prints the
bpw accounting; check it against the recipe:

```json
{ "params": 25598361600, "packed_bytes": 10871635968, "bpw_packed": 3.2408 }
```

Format details, including how to write your own reader: [FORMAT.md](FORMAT.md).

---

## 5. Stage 3 — verify

**Do this before you benchmark anything.** It takes about two seconds and no kernel build.

```bash
python -m prod verify /artifacts/medgemma-27b-cbk1
```

It reads the packed bytes back with the torch dequantizer and asserts:

* every blob is long enough for the arrays the manifest places in it;
* the bpw accounting recomputes from the per-matrix byte counts;
* the dequantized weights are finite and ~50% zero;
* for the 16:32 layouts, **every aligned 32-column block of every row has exactly 16
  survivors** — the fixed-offset decoder is not merely optimised for this, it is only
  correct when it holds;
* **every pruned position decodes to exactly 0.0.**

That last one is the assertion that matters, and it is worth understanding before you
write a reader of your own. An asymmetric quantizer stores a zero-point, so a pruned
position decodes as `(0 − zero) · scale` — which is *not* zero unless the packer placed
the grid's zero on a representable code. Get it wrong and you get a model that still
produces fluent text and quietly scores worse on everything. It cost us a real debugging
session in the DENSE4 path.

### Full kernel verification

```bash
python -m prod verify /artifacts/medgemma-27b-cbk1 \
    --config /models/medgemma-27b-text-it --kernel
```

Builds the extension and runs the megakernel against a pure-torch reference forward pass
(`src/cobaltkernel/ref_gemma3.py`) built by dequantizing **the same packed bytes**, so a
disagreement can only be a kernel bug, never a quantization difference. It reports:

* per-position max |logit difference| and argmax agreement over the whole prompt;
* 32 greedy decode steps, token for token;
* batch M=4 over four prompts against four independent M=1 runs (must be *bit-exact*);
* prefill-kernel → decode-kernel handoff through the shared KV cache.

Leave `--prompt-len` at its default of 1152: that is the length the bar below was
established at, and over 128 positions a single disagreement moves the figure by 0.8 pt.

**The bar is 98.5% argmax agreement, not 100%.** Two mathematically identical bf16
implementations of this model — HF eager vs HF sdpa — agree on only 98.61% of argmax
positions, because bf16 reduction order is not associative. Anything at or above ~98.5%
with a matching greedy token stream is a pass. If you see 100%, you are almost certainly
comparing something against itself.

---

## 6. Stage 4 — serve

```python
import sys; sys.path.insert(0, "/path/to/repo/src")
from prod import CobaltModel

m = CobaltModel.load_pretrained(
        "/artifacts/medgemma-27b-cbk1",
        config_dir="/models/medgemma-27b-text-it",
        max_ctx=2048)

print(m.generate_text("A 54-year-old presents with crushing chest pain and",
                      max_new_tokens=128))
```

or from the shell:

```bash
python -m prod generate /artifacts/medgemma-27b-cbk1 \
    --config /models/medgemma-27b-text-it \
    --prompt "A 54-year-old presents with" -n 64
```

The recipe is inferred from the artifact manifest; pass `recipe=` to override.

* `m.prefill(ids)` — the whole prompt in one cooperative launch, returns logits.
* `m.step(tokens, positions)` — one decode launch; token ids travel in the kernel
  parameter block, so a decode step performs *no* host-to-device copy at all.
* `m.generate(ids, n)` — the two composed, greedy.
* `m.stats` — recipe, bpw, bytes streamed per token, grid size.

**Batching.** `batch=M` selects the in-kernel batch, `M ∈ {1, 2, 4, 8}`. It is a
compile-time template parameter, so changing it triggers a rebuild. M=1 is where this
design wins; see §8.

---

## 7. Stage 5 — benchmark

```bash
python -m prod bench /artifacts/medgemma-27b-cbk1 \
    --config /models/medgemma-27b-text-it \
    --prompt 512 --gen 128 --out bench.json
```

Protocol: a 512-token prompt through the prefill kernel, then 128 tokens through the
decode kernel, greedy, batch 1. `decode_tok_s` covers tokens 2..128 only, so TTFT is
excluded and the number is a like-for-like decode rate.

The output ends with a comparison against what we measured for the same recipe. On
different hardware the absolute tok/s will differ — decode is bandwidth-bound, so it
tracks your card's memory bandwidth — but the **ratios between arms** should reproduce.
The comparison withholds ratios entirely if you benchmark a different model than the
reference (e.g. a 4B smoke test against the 27B reference), rather than printing a
flattering number that means nothing.

### Two measurement rules, learned expensively

1. **Benchmark the real kernel, never a proxy.** A standalone GEMV benchmark is
   throughput-bound; a decode step runs 0.9–3 waves and is *latency*-bound. Extrapolating
   the first to the second overpredicted decode by 20–28% here, twice, in the same
   direction, and once commissioned an entire format redesign that a later measurement
   killed.
2. **Never screen variants on a smaller GPU than you report on.** Halving the SM count
   changes warps per SM and can invert the ranking: a build that placed *first* on a
   1g slice placed *last* on the 2g slice it would actually ship on.

Also worth knowing: repeated identical runs of the per-phase breakdown differ by about
**2.4%**, so sub-2% per-phase deltas are not resolvable. End-to-end tok/s is much more
stable; trust that and treat the phase table as attribution, not measurement.

---

## 8. What to expect, and where this design does not win

MedGemma-27B, one 2g.48gb slice, 512→128, batch 1, against llama.cpp Q4_K_M measured on
the same slice under the same protocol:

| | Q4_K_M | `blk1632_b4` | `blk1632_b4_oproj` |
|---|---|---|---|
| weights | 16.55 GB | 10.95 GB | 11.12 GB |
| decode | 40.10 tok/s | **46.92** | 46.74 |
| TTFT (512 tok) | 237 ms | 329 ms | 330 ms |
| medical avg | 60.14 | 62.25 | **63.59** |

Being straight about the limits:

* **The advantage is a batch-1 advantage.** At M=32 our batched throughput is about 0.90×
  the DENSE4 control's and the weight-byte advantage compresses, because the KV cache term
  (0.484 MiB per token, identical for every arm) comes to dominate. This is a
  single-stream / low-concurrency design; for a high-throughput server, it is not the
  right tool.
* **Prefill is not the strong suit.** TTFT is ~1.4× llama.cpp's. The prefill kernel is a
  real batched `mma.sync` path (it went from 13.9 s to 0.33 s during development), but it
  has had far less tuning than decode.
* **Perplexity regresses** even where downstream task accuracy improves: 16.6 vs 11.6
  wikitext for Q4_K_M. The medical and ARC-easy numbers are what we optimised and what we
  report; the perplexity number is in the table because leaving it out would be dishonest.
* **The 27B target is Gemma3-architecture text models.** The reference forward pass,
  the RoPE tables and the sliding-window attention pattern are Gemma3-specific. Porting to
  another architecture is real work — see [KERNELS.md](KERNELS.md) §5.

---

## 9. Troubleshooting

> **The one that wastes the most time: a stale build lock looks exactly like a hang.**
> If a JIT build is interrupted — Ctrl-C, a timeout, a killed shell — torch leaves a
> zero-byte `lock` file in its build directory. The *next* run then blocks on that baton
> forever: the process is alive, the log is silent, and it holds **zero GPU memory**, so
> `nvidia-smi` shows nothing at all. It is indistinguishable from a slow compile.
>
> ```bash
> find "$TORCH_EXTENSIONS_DIR" -name lock -delete     # then re-run
> ```
>
> The tell is a live PID with no `nvcc` child and no new `.o` files. We have hit this
> ourselves more than once, including while preparing this guide.

| symptom | cause and fix |
|---|---|
| `Ninja is required to load C++ extensions` | `pip install ninja`, and put the venv's `bin/` on `PATH` — torch invokes the binary by name |
| `incompatible redefinition for option 'std'` | you passed `--std` to the extension; torch already appends `-std=c++20`. Drop yours |
| build fails on `cudaDeviceProp::clockRate` | CUDA 13 removed several `cudaDeviceProp` fields. Use `cudaDeviceGetAttribute(cudaDevAttrClockRate)` etc. |
| illegal memory access at launch, no compile error | an `extern __shared__` kernel launched with **0 bytes** of dynamic shared memory. Always request ≥ 4 B |
| model loads, output is fluent but wrong-ish | a diagnostic knob is exported in your shell. `env | grep COBALT` and clear it — or just go through `prod`, which clears them for you |
| verify reports "pruned positions do not decode to exactly zero" | the packer put the quantization grid's zero off-grid; do not deploy this artifact |
| decode much slower than expected, `nvidia-smi` shows nothing | on a shared machine, a killed `nvcc` can leave a stale torch `FileBaton`; jobs then block on the lock with **zero GPU memory** and are invisible to `nvidia-smi`. Check `TORCH_EXTENSIONS_DIR` for stale `.lock` files |
| first run takes ~60 s before anything happens | that is the JIT build. It is cached; subsequent runs are instant. Each distinct recipe compiles its own build directory |

---

## 10. End-to-end, copy-pasteable

```bash
export PYTHONPATH=/path/to/repo/src
export COBALT_CACHE=/big/volume/cobalt-cache        # optional, if $HOME is small

python -m prod doctor                                # 1. is the machine ready?
python -m prod recipes                               # 2. pick an arm

setsid nohup python -m prod quantize \
    -r blk1632_b4_oproj \
    --model /models/medgemma-27b-text-it \
    --out   /artifacts/mg27b-raw \
    > quant.log 2>&1 < /dev/null & disown            # 3. hours

python -m prod pack \
    -r blk1632_b4_oproj \
    --raw   /artifacts/mg27b-raw \
    --out   /artifacts/mg27b-cbk1 \
    --model /models/medgemma-27b-text-it             # 4. minutes

python -m prod verify /artifacts/mg27b-cbk1          # 5. seconds -- do not skip
python -m prod verify /artifacts/mg27b-cbk1 \
    --config /models/medgemma-27b-text-it --kernel   # 6. minutes, first build

python -m prod bench  /artifacts/mg27b-cbk1 \
    --config /models/medgemma-27b-text-it            # 7. the number
```
