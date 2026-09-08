#!/usr/bin/env bash
# attempt-7 matched HELD-OUT grid: does cobalt-awclipz beat CLIP-EQUIPPED baselines on 3 archs?
# All arms sp0.5 3-bit group-128 --force-true-bits (bpw-matched). awclipz given to baselines too
# (wanda-awq-awclipz, wanda-sinq-awclipz) = the critic's required fairness. cobalt & baselines RTN
# included as references (shows the clip lever's effect on each). Full-split downstream (real S3 test).
set -u
cd /workspace/PTQResearch
source env.sh >/dev/null 2>&1
CSV=/workspace/PTQResearch/results/awclipz_grid/results.csv
mkdir -p "$(dirname "$CSV")"
MODELS="gemma-2b tinyllama qwen-1.5b"
METHODS="cobalt cobalt-awclip wanda-awq wanda-awq-awclip wanda-sinq wanda-sinq-awclip"
DS="arc_easy,piqa,hellaswag,winogrande"
for M in $MODELS; do
  for MET in $METHODS; do
    echo "=== $M / $MET ==="
    python -u src/camera_bench.py --model "$M" --method "$MET" --sparsity 0.5 --bits 3 \
       --cobalt-group-size 128 --cobalt-beta 0.5 --force-true-bits \
       --csv "$CSV" --ppl-tasks wikitext2 --ds-tasks "$DS" --hp "g128" \
       || echo "CELL_ERR $M $MET"
  done
done
echo "AWCLIPZ_GRID_DONE"
