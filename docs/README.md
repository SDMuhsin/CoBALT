# CoBALT kernels — partner documentation

Custom CUDA inference kernels for CoBALT-compressed Gemma3 text models, plus the
quantization pipeline that produces the weights they read.

| document | read it for |
|---|---|
| `RUN.md` (prebuilt packages only) | **start here if you received weights** — the four commands, nothing else |
| [REPRODUCTION.md](REPRODUCTION.md) | set up, build, verify and benchmark the kernels on your own GPU |
| [FORMAT.md](FORMAT.md) | the CBK1 weight format: bit layout, bpw accounting, how to write your own reader |
| [KERNELS.md](KERNELS.md) | how the megakernel is put together, which source does what, what to tune when porting |

## Which of the two paths are you on?

**A — you received a prebuilt package** (a directory with `model/` in it, or a tarball of
one). The weights are already built: **skip stages 1 and 2 entirely.** You do not need a
Hugging Face checkpoint, calibration data, or the hours-long quantization run. Read the
package's `RUN.md`, then [REPRODUCTION.md](REPRODUCTION.md) §1 (prerequisites), §5
(verify), §6 (serve), §7 (benchmark) and §8 (where this design does not win).

**B — you are building an artifact from a Hugging Face checkpoint yourself.** Read
[REPRODUCTION.md](REPRODUCTION.md) start to finish. Note §3: the calibration file the
recipes pin is a research artifact that is **not in this repository**, so stage 1 cannot
run until you supply calibration text of your own.

Path A is the right one for reproducing our published numbers, because it removes both
the calibration data and the quantizer from the things that could differ between us.

## What this is, in one paragraph

The model is compressed by pruning half of every weight matrix and quantizing what
survives to 4 or 6 bits, then served by a **single persistent CUDA kernel** that does an
entire transformer forward step in one cooperative launch — no per-layer dispatch, no
cuBLAS, no host round-trip inside a decode step. On one MIG 2g.48gb slice of an RTX PRO
6000 Blackwell, MedGemma-27B decodes at **46.9 tok/s from an 11.0 GB artifact**, against
40.1 tok/s from 16.6 GB for llama.cpp Q4_K_M on the same slice under the same protocol —
1.17× the speed at 0.66× the weight bytes, with a slightly *higher* medical-benchmark
average than the bf16 model.

## The four shipped arms

| recipe | bpw | artifact | decode | medical avg | pick it when |
|---|---|---|---|---|---|
| `dense4` | 4.1875 | 14.2 GB | 41.8 tok/s | 62.47 | you want the control, or the simplest format |
| `blk1632_b4` | 3.1875 | 11.0 GB | **46.9 tok/s** | 62.25 | latency and footprint dominate |
| `blk1632_b4_oproj` | 3.2408 | 11.1 GB | 46.7 tok/s | **63.59** | best quality per byte — **the default recommendation** |
| `blk1632_b6_oproj4` | 4.1875 | 14.2 GB | 37.2 tok/s | **64.39** | best quality at DENSE4's exact byte count |

bf16 reference: 61.65 medical average from a 54 GB checkpoint.
`python -m prod recipes` prints this table with the full measurement protocol attached.
