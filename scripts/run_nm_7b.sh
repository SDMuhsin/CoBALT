#!/usr/bin/env bash
# 2:4 route at DEPLOYMENT scale (7B): is the 1B-class gate result about the regime or the method?
# 4-bit first (primary target), then 3-bit; same five arms as the smoke incl. cobalt-awclip (user steer).
# usage: scripts/run_nm_7b.sh <model> [MIG-UUID]
set -u; cd /workspace/PTQResearch
M=$1; [ -n "${2:-}" ] && export CUDA_VISIBLE_DEVICES=$2
source env.sh >/dev/null 2>&1
OUT=results/nm24/b7; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for B in 4 3; do
  for MTH in cobalt wanda-awq sparsegpt cobalt-awclip wanda-sinq; do
    EXTRA=""; [ "$MTH" = "sparsegpt" ] && EXTRA="--sgpt-percdamp 0.1"
    $PY src/camera_bench.py --model "$M" --method "$MTH" --nm 2:4 --bits $B --force-true-bits \
        --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 $EXTRA \
        > "$OUT/${M}_${MTH}_b${B}.log" 2>&1
    echo "[done] $M $MTH b$B"
  done
done
echo "ALLDONE_NM_7B_$M"
