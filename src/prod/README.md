# `prod` — the CoBALT deployment package

A production-facing wrapper around the research implementation in `../cobaltkernel/`.
Documentation for partners lives in [`../../docs/`](../../docs/); start with
`docs/REPRODUCTION.md`.

Two models ship through it: **MedGemma-27B** (four arms) and **BioMistral-7B** (one arm,
`blk1632_b4`, on the tensor-core decode build). The same commands serve both; nothing is
selected by hand — the layout comes from the artifact's `manifest.json` and the model from
the `config.json` beside the weights.

## The usual workflow

Most people never run stages 1 and 2. A clone has all the code and none of the data —
`.gitignore` excludes `results/`, and the weights were never in git — so the weights and
the calibration set ship separately as a tarball you unpack at the repo root. After that
the artifact is ready to use.

```bash
git clone <repo> && cd <repo>
tar -xf cobalt-biomistral-bigfiles.tar        # at the repo root; adds results/ and artifacts/
sha256sum -c SHA256SUMS

export PYTHONPATH=$PWD/src
python -m prod doctor                                       # is this machine ready?
python -m prod verify   artifacts/biomistral-7b-blk1632_b4  # ~2 s, no GPU
python -m prod verify   artifacts/biomistral-7b-blk1632_b4 --kernel   # the oracle gate
python -m prod generate artifacts/biomistral-7b-blk1632_b4 --prompt "..." -n 64
python -m prod bench    artifacts/biomistral-7b-blk1632_b4 --prompt 512 --gen 128
```

The MedGemma overlay (`cobalt-bigfiles.tar`) unpacks the same way to
`artifacts/medgemma-27b-<arm>`. `--config` is not needed for either: `config.json` and the
tokenizer sit beside the weights. The recipe comes from the artifact's `manifest.json` and
the model target from its `config.json`, so you cannot pair an artifact with the wrong
kernel configuration. `python -m prod recipes` prints the arms, `python -m prod targets`
the models and what each arm scored on them.

Two things that look like bugs and are not. The first `generate` spends 10–20 minutes in
nvcc with nothing on screen, because `megakernel.cu` is one big templated file; it is
cached afterwards. And if you ever kill a build partway, torch leaves a zero-byte `lock`
behind and the next run waits on it forever — alive, silent, using no GPU memory, so
`nvidia-smi` shows nothing wrong. Delete the lock under `TORCH_EXTENSIONS_DIR` and re-run.

Building an artifact yourself is stages 1 and 2 (`quantize`, `pack`). That path needs a
Hugging Face checkpoint and a calibration file, takes hours for a 27B model (about 6
minutes for BioMistral-7B), and is described in `docs/REPRODUCTION.md` §3–§4.

## Module map

| module | stage | what it is |
|---|---|---|
| `env.py` | — | toolchain and device configuration; `doctor` preflight |
| `recipes.py` | — | **the shipped arms (`RECIPES`) and models (`TARGETS`), pinned end to end** |
| `quantize.py` | 1 | HF checkpoint → raw CoBALT artifact |
| `pack.py` | 2 | raw artifact → CBK1 packed artifact (the VRAM image) |
| `verify.py` | 3 | structural + numerical checks, then kernel vs torch oracle |
| `model.py` | 4 | `CobaltModel`: load, prefill, decode, generate |
| `bench.py` | 5 | decode throughput, phase breakdown, comparison to reference |
| `cli.py` | — | `python -m prod <command>` |
| `selftest.py` | — | fast end-to-end check of the package itself |
| `_bridge.py` | — | import plumbing to `../cobaltkernel/` |

## Recipes and targets

A **recipe** is an arm: quantizer flags, packer flags and the kernel knobs that belong to
the layout. It is model-independent — the same `blk1632_b4` recipe built the MedGemma and
the BioMistral artifacts, and `python -m prod quantize -r blk1632_b4 --model <any HF
snapshot of a supported family>` builds another.

A **target** is a model we ship: the per-model kernel tuning that sits on top of the
recipe's knobs (for BioMistral-7B: the grid size, the attention split, the tensor-core
decode of the L2-resident phases, tensor-core attention, the selector table and the
row-decoder lookahead — every one of them measured, see `docs/KERNELS.md` §6) and the
numbers each arm measured on it. `recipes.activate(recipe, model_cfg)` applies both. A
model with no target runs the recipe exactly as pinned, and `bench` then prints the
measurement without a ratio rather than a ratio against the wrong model.

## What this package adds, and what it deliberately does not

It contains **no copy** of the quantizer, the packer, the CUDA kernels or the reference
model. One implementation, one place to fix a bug. What it adds is:

1. **One supported entry point per stage**, instead of a dozen research scripts with
   overlapping flags.
2. **A pinned configuration per shipped arm and model.** The kernel has ~40 environment
   knobs, eleven of which are diagnostics that deliberately compute the wrong answer.
   `recipes.activate()` sets the pinned ones and unsets every diagnostic, so a stray
   export in a partner's shell cannot silently produce a fast, subtly wrong model.
3. **Hardware autodetection** in place of the hard-coded `sm_120a` and fixed MIG slice the
   research tree assumed.
4. **Checks you can run before trusting a build** — `verify` catches the packing bugs that
   otherwise present as "the model works but scores a bit worse", and `verify --kernel`
   holds the kernel against a torch oracle built from the same packed bytes.
5. **Honest comparison.** `bench.compare_to_reference` withholds ratios when the model
   being benchmarked is not one the reference was measured on.

## Adding an arm or a model

An arm: add a `Recipe` to `recipes.py` — quantizer flags, packer flags, kernel env — and
register its layout signature in `model._BY_LAYOUT` so artifacts infer it. A model: add a
`Target` keyed by its `config.json` `model_type` and `hidden_size`, with its tuning and
the numbers you measured; `python -m prod.selftest` checks that the tuning is a legal knob
set. Nothing else needs to change.
