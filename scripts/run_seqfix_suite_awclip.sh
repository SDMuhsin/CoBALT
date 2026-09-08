#!/usr/bin/env bash
# Wave 2 of the seqfix suite: the awclip pair (cobalt-awclip vs cobalt-seqfix-awclip) across families.
# Same grid as wave 1 (user steer: whatever is run for base CoBALT is run for CoBALT-awclip).
# usage: scripts/run_seqfix_suite_awclip.sh <model> [<model> ...]
set -u; cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=""; source env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for M in "$@"; do
  OUT=results/seqfix_suite/$M; mkdir -p "$OUT/logs"
  $VENV/bin/python scripts/camera_dispatch.py --model "$M" --force-true-bits --cobalt-group-size 128 \
      --col-balance-exps 0.5 --methods cobalt-awclip,cobalt-seqfix-awclip \
      --bits 3,4 --sparsities 0.4,0.5,0.6,0.7,0.8 \
      --ds-tasks arc_easy,piqa,hellaswag,winogrande --ppl-tasks wikitext2 \
      --csv "$OUT/results.csv" --log-dir "$OUT/logs" >> "$OUT/dispatch.log" 2>&1
  echo "MODEL_DONE $M"
done
echo "ALLDONE_SEQFIX_AWCLIP"
