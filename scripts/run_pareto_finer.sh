#!/bin/bash
# Finer Pareto sweep: sparsity is FREE on the memory axis for fixed 2-bit repack
# (3.25 bpw at every sp). Fill in sp{0.5,0.7} to locate the best operating point
# vs the 3-bit baselines at matched sparsity.
#   - 2-bit cobalt-repack  -> results/eout_2bit/results.csv   (3.25 bpw)
#   - 3-bit wanda-{awq,sinq}-repack -> results/eout_bench/results.csv (4.25 bpw)
# Resume-safe (self-skips settled cells).
set -u
cd "$(dirname "$0")/.."
source env.sh
for M in gemma-2b tinyllama qwen-1.5b; do
  echo "=== finer 2bit cobalt-repack model $M ==="
  python scripts/camera_dispatch.py \
    --model "$M" --methods cobalt-repack \
    --sparsities 0.5,0.7 --bits 2 \
    --ppl-tasks wikitext2 --ds-tasks arc_easy,piqa,hellaswag,winogrande \
    --csv results/eout_2bit/results.csv --log-dir results/eout_2bit/logs \
    --cobalt-group-size 128 --force-true-bits

  echo "=== finer 3bit baseline-repack model $M ==="
  python scripts/camera_dispatch.py \
    --model "$M" --methods wanda-awq-repack,wanda-sinq-repack \
    --sparsities 0.5,0.7 --bits 3 \
    --ppl-tasks wikitext2 --ds-tasks arc_easy,piqa,hellaswag,winogrande \
    --csv results/eout_bench/results.csv --log-dir results/eout_bench/logs \
    --cobalt-group-size 128 --force-true-bits
done
echo "ALL FINER PARETO DISPATCHED-AND-DONE"
