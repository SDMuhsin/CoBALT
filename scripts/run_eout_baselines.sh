#!/bin/bash
# math4 fairness: give repacking to the baselines. {wanda-awq,wanda-sinq} native +
# {wanda-awq-repack,wanda-sinq-repack} x {gemma-2b,tinyllama,qwen-1.5b} x sp{0.4,0.6,0.8} @3-bit.
# Into the SAME csv as cobalt/cobalt-repack so the analyzer compares all arms.
set -u
cd "$(dirname "$0")/.."
source env.sh
for M in gemma-2b tinyllama qwen-1.5b; do
  echo "=== baselines model $M ==="
  python scripts/camera_dispatch.py \
    --model "$M" \
    --methods wanda-awq,wanda-sinq,wanda-awq-repack,wanda-sinq-repack \
    --sparsities 0.4,0.6,0.8 --bits 3 \
    --ppl-tasks wikitext2 \
    --ds-tasks arc_easy,piqa,hellaswag,winogrande \
    --csv results/eout_bench/results.csv \
    --log-dir results/eout_bench/logs \
    --cobalt-group-size 128
done
echo "ALL BASELINES DISPATCHED-AND-DONE"
