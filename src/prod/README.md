# `prod` — the CoBALT deployment package

A production-facing wrapper around the research implementation in `../cobaltkernel/`.
Documentation for partners lives in [`../../docs/`](../../docs/); start with
`docs/REPRODUCTION.md`.

## The usual workflow

Most people never run stages 1 and 2. A clone has all the code and none of the data —
`.gitignore` excludes `results/`, and the weights were never in git — so the weights and
the calibration set ship separately as a tarball you unpack at the repo root. After that
the artifact is ready to use.

```bash
git clone <repo> && cd <repo>
tar -xf cobalt-bigfiles.tar       # at the repo root; adds results/ and artifacts/
sha256sum -c SHA256SUMS

export PYTHONPATH=$PWD/src
python -m prod doctor                                            # is this machine ready?
python -m prod verify   artifacts/medgemma-27b-blk1632_b4_oproj  # ~2 s, no GPU
python -m prod generate artifacts/medgemma-27b-blk1632_b4_oproj --prompt "..." -n 64
python -m prod bench    artifacts/medgemma-27b-blk1632_b4_oproj --prompt 512 --gen 128
```

`--config` is not needed: `config.json` and the tokenizer sit beside the weights. The
recipe comes from the artifact's `manifest.json`, so you cannot pair an artifact with the
wrong kernel configuration. `python -m prod recipes` prints the arms and what each scored.

Two things that look like bugs and are not. The first `generate` spends 10–20 minutes in
nvcc with nothing on screen, because `megakernel.cu` is one big templated file; it is
cached afterwards. And if you ever kill a build partway, torch leaves a zero-byte `lock`
behind and the next run waits on it forever — alive, silent, using no GPU memory, so
`nvidia-smi` shows nothing wrong. Delete the lock under `TORCH_EXTENSIONS_DIR` and re-run.

Building an artifact yourself is stages 1 and 2 (`quantize`, `pack`). That path needs a
Hugging Face checkpoint and a calibration file, takes hours for a 27B model, and is
described in `docs/REPRODUCTION.md` §3–§4.

## Module map

| module | stage | what it is |
|---|---|---|
| `env.py` | — | toolchain and device configuration; `doctor` preflight |
| `recipes.py` | — | **the four shipped arms, pinned end to end** |
| `quantize.py` | 1 | HF checkpoint → raw CoBALT artifact |
| `pack.py` | 2 | raw artifact → CBK1 packed artifact (the VRAM image) |
| `verify.py` | 3 | structural + numerical checks, then kernel vs torch oracle |
| `model.py` | 4 | `CobaltModel`: load, prefill, decode, generate |
| `bench.py` | 5 | decode throughput, phase breakdown, comparison to reference |
| `cli.py` | — | `python -m prod <command>` |
| `selftest.py` | — | fast end-to-end check of the package itself |
| `_bridge.py` | — | import plumbing to `../cobaltkernel/` |

## What this package adds, and what it deliberately does not

It contains **no copy** of the quantizer, the packer, the CUDA kernels or the reference
model. One implementation, one place to fix a bug. What it adds is:

1. **One supported entry point per stage**, instead of a dozen research scripts with
   overlapping flags.
2. **A pinned configuration per shipped arm.** The kernel has ~25 environment knobs, six
   of which are diagnostics that deliberately compute the wrong answer. `recipes.activate()`
   sets the pinned ones and unsets every diagnostic, so a stray export in a partner's
   shell cannot silently produce a fast, subtly wrong model.
3. **Hardware autodetection** in place of the hard-coded `sm_120a` and fixed MIG slice the
   research tree assumed.
4. **Checks you can run before trusting a build** — `verify` catches the packing bugs that
   otherwise present as "the model works but scores a bit worse".
5. **Honest comparison.** `bench.compare_to_reference` withholds ratios when the model
   being benchmarked is not the one the reference was measured on.

## Adding an arm

Add a `Recipe` to `recipes.py` — quantizer flags, packer flags, kernel env, and the
numbers you measured — and register its layout signature in `model._BY_LAYOUT` so
artifacts infer it. Nothing else needs to change.
