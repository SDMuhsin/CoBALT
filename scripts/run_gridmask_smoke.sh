#!/usr/bin/env bash
# Held-out 2-bit smoke for the grid-aware MASK (fresh commission). Tests whether the MEASURED
# calib finding (grid mask is CoBALT-specific + awclip-orthogonal at 2-bit) transfers to held-out
# downstream on gemma-2b. 4 arms at MATCHED 2-bit/sp0.5/g128, --force-true-bits:
#   cobalt                 = balanced mask + RTN            (base)
#   cobalt-awclip          = balanced mask + awclip         (strongest matched incumbent; universal lever)
#   cobalt-gridmask        = GRID mask     + RTN
#   cobalt-gridmask-awclip = GRID mask     + awclip         (the proposed co-design)
# S4/orthogonality question = does gridmask-awclip beat awclip (holding awclip fixed, swap the mask)?
set -u
cd "$(dirname "$0")/.."
source env.sh 2>/dev/null
CSV=results/gridmask_smoke/smoke.csv
mkdir -p results/gridmask_smoke
for M in cobalt cobalt-awclip cobalt-gridmask cobalt-gridmask-awclip; do
  echo "=== $M ==="
  python -u src/camera_bench.py --model gemma-2b --method "$M" --sparsity 0.5 --bits 2 \
    --cobalt-group-size 128 --force-true-bits \
    --ppl-tasks wikitext2 --ds-tasks arc_easy,piqa \
    --csv "$CSV" --hp "gridmask_smoke"
done
echo GRIDMASK_SMOKE_DONE
