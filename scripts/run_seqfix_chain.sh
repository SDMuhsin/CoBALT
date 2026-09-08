#!/usr/bin/env bash
# Serialize the seqfix suite: wave 1 (RTN pair) -> wave 2 (awclip pair) -> gap-fill sweep.
# ONE dispatcher at a time: two concurrent dispatchers race on the free-memory check and land 4 cells
# on one 24GB slice, which OOMs a cell -- and an OOM writes a FAILED row that the dispatcher then treats
# as SETTLED, silently dropping the cell from the grid. The final sweep re-enumerates all four arms so
# any cell dropped that way (or by a crash) is rebuilt.
# usage: scripts/run_seqfix_chain.sh <wait-pid> <model> [<model> ...]
set -u; cd /workspace/PTQResearch
WAITPID=$1; shift
while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
scripts/run_seqfix_suite_awclip.sh "$@"
export CUDA_VISIBLE_DEVICES=""; source env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for M in "$@"; do
  OUT=results/seqfix_suite/$M; mkdir -p "$OUT/logs"
  $VENV/bin/python scripts/camera_dispatch.py --model "$M" --force-true-bits --cobalt-group-size 128 \
      --col-balance-exps 0.5 --methods cobalt,cobalt-seqfix,cobalt-awclip,cobalt-seqfix-awclip \
      --bits 3,4 --sparsities 0.4,0.5,0.6,0.7,0.8 --free-gb 20 \
      --ds-tasks arc_easy,piqa,hellaswag,winogrande --ppl-tasks wikitext2 \
      --csv "$OUT/results.csv" --log-dir "$OUT/logs" >> "$OUT/dispatch.log" 2>&1
  echo "SWEEP_DONE $M"
done
echo "ALLDONE_SEQFIX_CHAIN"
