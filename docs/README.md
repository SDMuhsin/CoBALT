# CoBALT kernels — partner documentation

Custom CUDA inference kernels for CoBALT-compressed Gemma3 (MedGemma-27B) and
Llama-family (BioMistral-7B / Mistral-7B) text models, plus the quantization pipeline that
produces the weights they read.

| document | read it for |
|---|---|
| [REPRODUCTION.md](REPRODUCTION.md) | **start here** — set up, verify, benchmark and serve on your own GPU |
| [FORMAT.md](FORMAT.md) | the CBK1 weight format: bit layout, bpw accounting, how to write your own reader |
| [KERNELS.md](KERNELS.md) | how the megakernel is put together, which source does what, what to tune when porting |

## The repository is not self-sufficient — you also need the big-files overlay

`.gitignore` excludes `results/`, and the compressed weights were never in git. So a
clone gives you all of the code and documentation and **none** of the data. We ship the
rest as a separate **drop-in overlay per model** — `cobalt-bigfiles.tar` for MedGemma-27B,
`cobalt-biomistral-bigfiles.tar` for BioMistral-7B: unpack it at the root of your clone
and every path the code already uses resolves. Both can live in one clone.

```bash
git clone <repo> && cd <repo>
tar -xf cobalt-biomistral-bigfiles.tar   # adds results/ and artifacts/, overwrites nothing
```

It carries the packed weights (with `config.json` and the tokenizer beside them) and the
calibration set the recipes pin. It deliberately carries no code and no docs — those are
the clone's job, so the two can never drift apart.

With the overlay in place the weights are already built, so **stages 1 and 2 of
[REPRODUCTION.md](REPRODUCTION.md) are already done**: no Hugging Face download, no
calibration pass, no hours-long quantization run. Go to §0, then §5–§8.

Without it, `python -m prod quantize` fails immediately on the missing calibration file —
see [REPRODUCTION.md](REPRODUCTION.md) §3 for how to substitute your own.

## What this is, in one paragraph

The model is compressed by pruning half of every weight matrix and quantizing what
survives to 4 or 6 bits, then served by a **single persistent CUDA kernel** that does an
entire transformer forward step in one cooperative launch — no per-layer dispatch, no
cuBLAS, no host round-trip inside a decode step. On one MIG 2g.48gb slice of an RTX PRO
6000 Blackwell, MedGemma-27B decodes at **46.9 tok/s from an 11.0 GB artifact**, against
40.1 tok/s from 16.6 GB for llama.cpp Q4_K_M on the same slice under the same protocol —
1.17× the speed at 0.66× the weight bytes, with a slightly *higher* medical-benchmark
average than the bf16 model.

## The shipped models and arms

**BioMistral-7B** — one arm, `blk1632_b4`, served by the tensor-core decode build. One
2g.48gb MIG slice of an RTX PRO 6000 Blackwell, 512→128, batch 1, llama.cpp interleaved
in the same window and pinned to the same cores, mean of two passes:

| | llama.cpp Q4_K_M (imatrix) | `blk1632_b4` |
|---|---|---|
| bpw / artifact | 4.8 / 4.37 GB | **3.19 / 2.92 GB** |
| decode | 143.8 tok/s | **164.3 tok/s (1.14×)** |
| TTFT, 512-token prompt | 77 ms | 112 ms |
| medical avg (MedQA, PubMedQA, MedMCQA) | 53.23 | 49.72 |
| wikitext PPL | 13.57 | 15.47 |

bf16 reference: 53.47 medical average. The speed and byte wins carry over from MedGemma;
the quality does not — on this 7B model the arm sits **3.5 pt below Q4_K_M** on the
medical average, and the package ships that artifact as measured. See
[REPRODUCTION.md](REPRODUCTION.md) §8.

**MedGemma-27B** — four arms:

| recipe | bpw | artifact | decode | medical avg | pick it when |
|---|---|---|---|---|---|
| `dense4` | 4.1875 | 14.2 GB | 41.8 tok/s | 62.47 | you want the control, or the simplest format |
| `blk1632_b4` | 3.1875 | 11.0 GB | **46.9 tok/s** | 62.25 | latency and footprint dominate |
| `blk1632_b4_oproj` | 3.2408 | 11.1 GB | 46.7 tok/s | **63.59** | best quality per byte — **the default recommendation** |
| `blk1632_b6_oproj4` | 4.1875 | 14.2 GB | 37.2 tok/s | **64.39** | best quality at DENSE4's exact byte count |

bf16 reference: 61.65 medical average from a 54 GB checkpoint.
`python -m prod recipes` prints the arms and `python -m prod targets` the models, each
with the full measurement protocol attached.
