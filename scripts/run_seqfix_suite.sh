#!/usr/bin/env bash
# SEQFIX SUITE (2026-09-03): does CoBALT-seqfix (sequential compressed-prefix inputs, everything else
# identical to the submitted CoBALT) consistently beat baseline CoBALT? Head-to-head at MATCHED beta,
# matched bits/group/sparsity/calibration, across families x bits {3,4} x sparsity {0.4..0.8} x 4 full-split
# reasoning tasks + wikitext2 PPL. Both the RTN and the awclip quantizer stage are run for BOTH arms
# (user steer: whatever is done for base CoBALT is done for CoBALT-awclip).
# usage: scripts/run_seqfix_suite.sh <model> [<model> ...]
set -u; cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=""; source env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for M in "$@"; do
  OUT=results/seqfix_suite/$M; mkdir -p "$OUT/logs"
  $VENV/bin/python scripts/camera_dispatch.py --model "$M" --force-true-bits --cobalt-group-size 128 \
      --col-balance-exps 0.5 \
      --methods cobalt,cobalt-seqfix,cobalt-awclip,cobalt-seqfix-awclip \
      --bits 3,4 --sparsities 0.4,0.5,0.6,0.7,0.8 \
      --ds-tasks arc_easy,piqa,hellaswag,winogrande --ppl-tasks wikitext2 \
      --csv "$OUT/results.csv" --log-dir "$OUT/logs" > "$OUT/dispatch.log" 2>&1
  echo "MODEL_DONE $M"
done
echo "ALLDONE_SEQFIX_SUITE"
