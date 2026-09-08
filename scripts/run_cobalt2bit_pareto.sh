#!/bin/bash
# Pareto probe: 2-bit CoBALT (repacked, ~3.25 bpw) vs the EXISTING 3-bit sparse
# baseline-repack arms (~4.25 bpw, already in results/eout_bench/results.csv).
# If 2-bit cobalt-repack matches/beats the 3-bit baselines at LOWER bpw, the
# OBS-lever fairness objection dissolves (win at strictly lower compression).
# Only the 2-bit cobalt-repack arm is new; baselines are reused from the 3-bit grid.
set -u
cd "$(dirname "$0")/.."
source env.sh
for M in gemma-2b tinyllama qwen-1.5b; do
  echo "=== 2bit cobalt-repack model $M ==="
  python scripts/camera_dispatch.py \
    --model "$M" \
    --methods cobalt-repack \
    --sparsities 0.4,0.6,0.8 --bits 2 \
    --ppl-tasks wikitext2 \
    --ds-tasks arc_easy,piqa,hellaswag,winogrande \
    --csv results/eout_2bit/results.csv \
    --log-dir results/eout_2bit/logs \
    --cobalt-group-size 128 \
    --force-true-bits
done
echo "ALL 2BIT COBALT DISPATCHED-AND-DONE"
