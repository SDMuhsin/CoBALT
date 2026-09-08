#!/usr/bin/env bash
# 2:4 route, GATE 1 (regime): one model, bits {3,4}, 2:4 for EVERY arm (cobalt, cobalt-awclip, and the
# strongest matched baselines wanda-awq / wanda-sinq / tuned SparseGPT pd=0.1). Pre-registered kill:
# cobalt ~= wanda-* at both bit-widths on both smoke models => direction dies, no grid.
# usage: scripts/run_nm_smoke.sh <model> [MIG-UUID]
set -u; cd /workspace/PTQResearch
M=$1; [ -n "${2:-}" ] && export CUDA_VISIBLE_DEVICES=$2
source env.sh >/dev/null 2>&1
OUT=results/nm24/smoke; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for B in 3 4; do
  for MTH in cobalt cobalt-awclip wanda-awq wanda-sinq sparsegpt; do
    EXTRA=""; [ "$MTH" = "sparsegpt" ] && EXTRA="--sgpt-percdamp 0.1"
    $PY src/camera_bench.py --model "$M" --method "$MTH" --nm 2:4 --bits $B --force-true-bits \
        --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 $EXTRA \
        > "$OUT/${M}_${MTH}_b${B}.log" 2>&1
    echo "[done] $M $MTH b$B"
  done
done
echo "ALLDONE_NM_SMOKE_$M"
