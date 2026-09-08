# CoBALT kernels — partner documentation

Custom CUDA inference kernels for CoBALT-compressed Gemma3 text models, plus the
quantization pipeline that produces the weights they read.

| document | read it for |
|---|---|
| [REPRODUCTION.md](REPRODUCTION.md) | **start here** — set up, build, verify and benchmark the kernels on your own GPU |
| [FORMAT.md](FORMAT.md) | the CBK1 weight format: bit layout, bpw accounting, how to write your own reader |
| [KERNELS.md](KERNELS.md) | how the megakernel is put together, which source does what, what to tune when porting |

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
