#!/usr/bin/env bash
# Bit-MATCHED hyperparameter grid-search on electra-base-mnli (3-bit + 4-bit, sp 0.4..0.8).
# Sweeps only bpw-PRESERVING knobs so every arm stays matched:
#   cobalt beta {0..1 step .1} | sparsegpt percdamp{1e-3,1e-2,1e-1} x blocksize{64,128}
#   wanda-awq / wanda-sinq: no exposed matched knob -> run at default (anchors)
# Fans one (bits,sparsity,method) job per FREE MIG slice; each job loops its hp grid
# internally. Resumable, requeues transient failures. Output: results/benchmark_hpgrid_electra/.
#
# Launch (survives disconnect):
#     mkdir -p results/benchmark_hpgrid_electra
#     nohup bash scripts/run_hpgrid.sh > results/benchmark_hpgrid_electra/dispatch.log 2>&1 &
# Preview only:
#     bash scripts/run_hpgrid.sh --dry-run
set -u
cd "$(dirname "$0")/.."
# The scheduler itself uses no GPU; each job subprocess pins its own MIG slice.
export CUDA_VISIBLE_DEVICES=""
source env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python scripts/dispatch_hpgrid.py "$@"
