#!/bin/bash
# math4 empirical grid: 3 models x {fp16,cobalt,cobalt-repack,cobalt-eout} x sp{0.4,0.6,0.8} @ 3-bit
# wikitext2 PPL + arc_easy/piqa/hellaswag/winogrande. MIG-parallel, resume-safe.
set -u
cd "$(dirname "$0")/.."
source env.sh
mkdir -p results/eout_bench
for M in gemma-2b tinyllama qwen-1.5b; do
  echo "=== dispatching model $M ==="
  python scripts/camera_dispatch.py \
    --model "$M" \
    --methods fp16,cobalt,cobalt-repack,cobalt-eout \
    --sparsities 0.4,0.6,0.8 --bits 3 \
    --ppl-tasks wikitext2 \
    --ds-tasks arc_easy,piqa,hellaswag,winogrande \
    --csv results/eout_bench/results.csv \
    --log-dir results/eout_bench/logs \
    --cobalt-group-size 128
done
echo "ALL MODELS DISPATCHED-AND-DONE"
