#!/usr/bin/env bash
# 2-bit + sparsity{0.4..0.8} MATCHED quant+prune sweep, CoBALT vs best-of-SOTA.
# Fans one (harness,model,method,sparsity) CELL per FREE MIG slice, resumable.
#
#   roberta-large-mnli / electra-base-mnli / roberta-base-mnli  (MNLI acc)
#   dinov2-large                                                (ImageNet top-1, 10k)
#   gemma-2b                                                    (arc_easy/hellaswag/piqa/winogrande)
#
# Launch (survives disconnect):
#     nohup bash scripts/run_2bit_sweep.sh > results/benchmark_2bit/dispatch.log 2>&1 &
# Preview only:
#     bash scripts/run_2bit_sweep.sh --dry-run
set -u
cd "$(dirname "$0")/.."
# The scheduler itself uses no GPU; each cell subprocess pins its own MIG slice.
export CUDA_VISIBLE_DEVICES=""
source env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# bit-width (and thus output dir results/benchmark_{bits}bit/) is set via --bits; dispatcher mkdirs it.
exec python scripts/dispatch_2bit.py "$@"
