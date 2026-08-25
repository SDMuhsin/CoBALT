#!/usr/bin/env bash
# Full bit-matched TUNED GLUE suite for a chosen encoder family (roberta-large | deberta-v3-base).
# Same 27-hp-cell/task grid as the electra suite. Checkpoints must be dense-verified & cached first.
#
# Launch (survives disconnect), e.g. roberta-large:
#     mkdir -p results/benchmark_hpgrid_roberta-large_glue
#     nohup bash scripts/run_glue_suite_multi.sh --family roberta-large \
#         > results/benchmark_hpgrid_roberta-large_glue/dispatch.log 2>&1 &
# Preview:
#     bash scripts/run_glue_suite_multi.sh --family deberta-v3-base --dry-run
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
source env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python scripts/dispatch_glue_suite_multi.py "$@"
