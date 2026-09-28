# CoBALT: A Column-Balanced Mask for Sparse Low-Bit Compression of Language Models

**Status:** Under review at Elsevier *Neurocomputing*.
For a preprint, contact the author at **sdmuhsin@gmail.com**.

## Abstract

We present CoBALT (Column-Balanced Thresholding), a post-training method that
compresses each linear layer of a trained network into a weight matrix that is
at once sparse and low-bit. CoBALT runs three stages on one layer at a time. A
column-balanced quantile mask, the one part CoBALT introduces, selects the
weights to keep by balancing a Wanda importance score across both matrix axes
before a single global threshold. An optimal-brain-surgeon update then moves the
pruned mass onto the surviving weights, and a group-wise round-to-nearest
quantizer maps them to the target bit-width. On a decoder and three fine-tuned
encoders CoBALT holds the best task-mean downstream accuracy of any matched
method through most of a joint pruning and quantization grid, and its lead is
widest at low bit width and mid-range sparsity. On a vision transformer the same
mask helps under light pruning and loses accuracy under heavy pruning. The
method is aimed at low-bit compression of language models.

## Deploying it: CUDA kernels and compressed weights

Separately from the paper, this repository carries a CUDA inference stack that serves a
CoBALT-compressed **MedGemma-27B** in one cooperative kernel launch per step, and the
packaging around it. If you are here to *run* a compressed model rather than to read the
method, start at **[docs/REPRODUCTION.md](docs/REPRODUCTION.md)**.

| | |
|---|---|
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | set up, verify, benchmark and serve on your own GPU |
| [docs/FORMAT.md](docs/FORMAT.md) | the CBK1 weight format |
| [docs/KERNELS.md](docs/KERNELS.md) | megakernel internals and what to tune when porting |

**A clone is not self-sufficient.** `.gitignore` excludes `results/`, and the compressed
weights were never in git, so the code and docs are all here but none of the data is. The
weights and the calibration set ship separately as a drop-in overlay that you unpack at
the repo root — see [docs/README.md](docs/README.md).

## Method source

- `src/nosink.py` implements the CoBALT compression of one linear layer: the
  column-balanced quantile mask (`--mask balanced --col-balance-exp`), the
  optimal-brain-surgeon compensation, and the group-wise round-to-nearest
  quantizer.
- `src/camera_bench.py` is the benchmark entry point that applies the method to
  a model and evaluates it. CoBALT is selected with `--method cobalt`; the
  column-balance exponent is the tuned, bit-budget-preserving knob.

A single-cell run:

```bash
python src/camera_bench.py --method cobalt --sparsity 0.7 --bits 3
```

## Contact

Sayed Muhsin, sdmuhsin@gmail.com
