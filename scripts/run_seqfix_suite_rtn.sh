#!/usr/bin/env bash
# Wave 1 of the seqfix suite: the RTN pair (cobalt vs cobalt-seqfix) across families.
# usage: scripts/run_seqfix_suite_rtn.sh <model> [<model> ...]
set -u; cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=""; source env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for M in "$@"; do
  OUT=results/seqfix_suite/$M; mkdir -p "$OUT/logs"
  $VENV/bin/python scripts/camera_dispatch.py --model "$M" --force-true-bits --cobalt-group-size 128 \
      --col-balance-exps 0.5 --methods cobalt,cobalt-seqfix \
      --bits 3,4 --sparsities 0.4,0.5,0.6,0.7,0.8 \
      --ds-tasks arc_easy,piqa,hellaswag,winogrande --ppl-tasks wikitext2 \
      --csv "$OUT/results.csv" --log-dir "$OUT/logs" >> "$OUT/dispatch.log" 2>&1
  echo "MODEL_DONE $M"
done
echo "ALLDONE_SEQFIX_RTN"
