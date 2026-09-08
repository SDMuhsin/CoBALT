# `prod` — the CoBALT deployment package

A production-facing wrapper around the research implementation in `../cobaltkernel/`.
Documentation for partners lives in [`../../docs/`](../../docs/); start with
`docs/REPRODUCTION.md`.

```bash
export PYTHONPATH=/path/to/repo/src
python -m prod doctor          # is this machine ready?
python -m prod recipes         # what can I build, and what should it score?
```

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
