#!/usr/bin/env bash
# CoBALT-16:32 (fixed-cardinality mask) on the gemma-2b four-task grid at s=0.50.
# Matched to the `cobalt` arm: same bits, same group size, same calibration, same beta grid.
set -u
cd "$(dirname "$0")/.."
source env.sh >/dev/null 2>&1
PY=/scratch/root/PTQResearch/env/bin/python
CSV=results/blk_grid/results.csv
mkdir -p results/blk_grid
for BITS in 3 4; do
  for BETA in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "=== bits=$BITS beta=$BETA $(date +%H:%M:%S) ==="
    PYTHONUNBUFFERED=1 $PY -u src/camera_bench.py \
      --model gemma-2b --method cobalt-blk32 --bits "$BITS" --sparsity 0.50 \
      --cobalt-beta "$BETA" --cobalt-group-size 128 --hp "beta=$BETA" \
      --ds-tasks arc_easy,piqa,hellaswag,winogrande --ppl-tasks "" \
      --csv "$CSV" 2>&1 | tail -3
  done
done
echo "BLK32_GRID_DONE $(date +%H:%M:%S)"
