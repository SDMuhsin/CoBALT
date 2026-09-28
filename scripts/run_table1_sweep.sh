#!/bin/bash
# Re-run the paper's tuned grids with every arm swept over a grid at least as large as
# CoBALT's (src/tuned_grids.py). Resume-safe: cells already settled in the CSV are skipped,
# so this can be killed and relaunched. Launch detached:
#   setsid nohup bash scripts/run_table1_sweep.sh gemma > <log> 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
source env.sh
unset CUDA_VISIBLE_DEVICES      # the dispatcher places cells on slices itself

SUITE="${1:-gemma}"
# Matched joint baselines first (they carry the headline comparison), anchors last.
METHODS="wanda-awq,wanda-sinq,sparsegpt,jsq-wo,slim,cobalt,awq,sinq,wanda,fp16"
# Composite arms (CoBALT mask x another method's quantizer/recipe) + the mirror control.
# Appended, not merged, so the Table-1 baseline set above stays exactly what it was.
COMPOSITES="cobalt-sinq,cobalt-awq,cobaltmask-sgpt,sgptmask-cobalt"
[ "${WITH_COMPOSITES:-1}" = 1 ] && METHODS="$METHODS,$COMPOSITES"

case "$SUITE" in
  gemma)
    exec python scripts/camera_dispatch.py --tuned \
      --model gemma-2b --methods "$METHODS" \
      --bits 3,4 --sparsities 0.4,0.5,0.6,0.7,0.8 \
      --ds-tasks arc_easy,piqa,hellaswag,winogrande --ppl-tasks "" \
      --cobalt-group-size 128 --force-true-bits \
      --csv results/benchmark_camera_ds_tuned/results.csv \
      --log-dir results/benchmark_camera_ds_tuned/logs
    ;;
  vit)
    exec python scripts/dispatch_vit_suite.py
    ;;
  glue)
    exec python scripts/dispatch_glue_suite_multi.py "${@:2}"
    ;;
  *)
    echo "usage: $0 {gemma|vit|glue}" >&2; exit 2 ;;
esac
