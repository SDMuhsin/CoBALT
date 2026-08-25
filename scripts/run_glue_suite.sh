#!/usr/bin/env bash
# Full bit-matched TUNED suite: all matched methods x all GLUE tasks on electra-base (3b+4b, sp .4-.8).
# 7 tasks (mnli,sst2,cola,mrpc,qnli,qqp,rte), each with its own dense-verified electra checkpoint.
# Fans one (task,bits,sparsity,method) job per FREE MIG slice; each job loops its hp grid internally.
# Resumable, requeues failures. Output: results/benchmark_hpgrid_electra_glue/.
#
# Checkpoints/datasets are already cached (downloaded + dense-verified), so this runs OFFLINE.
# Launch (survives disconnect):
#     mkdir -p results/benchmark_hpgrid_electra_glue
#     nohup bash scripts/run_glue_suite.sh > results/benchmark_hpgrid_electra_glue/dispatch.log 2>&1 &
# Preview only:
#     bash scripts/run_glue_suite.sh --dry-run
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
source env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python scripts/dispatch_glue_suite.py "$@"
