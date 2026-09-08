#!/usr/bin/env bash
# SMOKE (attempt-11 #21 hard column-floor mask): ONE operating point (sp0.7, 3-bit) on the two core
# healthy models, with the S4 attribution controls. base cobalt vs cobalt-floor{0.25,0.5} vs strongest
# baseline (wanda-awq) vs wanda-floor (is the floor universal?). arc_easy+piqa (limit 1000) + wiki PPL.
# STOP for a blind critic after this -- NO grid.
set -u
cd /workspace/PTQResearch
source env.sh >/dev/null 2>&1
SP="${1:-0.7}"
OUT=results/maskbite/floor_smoke_sp${SP}
mkdir -p "$OUT"
CSV="$OUT/results.csv"
PY=$VENV/bin/python
BITS=3; LIMIT=1000
run () { # model method extra_label floor
  local M=$1 MTH=$2 FF=$3
  $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity $SP --bits $BITS \
      --force-true-bits --cobalt-floor-frac "$FF" --limit $LIMIT --csv "$CSV" \
      --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 --hp "floor=$FF" \
      > "$OUT/${M}_${MTH}_ff${FF}.log" 2>&1
  echo "[done] $M $MTH ff$FF"
}
for M in tinyllama qwen-1.5b; do
  run "$M" cobalt 0.0
  run "$M" cobalt-floor 0.25
  run "$M" cobalt-floor 0.5
  run "$M" wanda-awq 0.0
  run "$M" wanda-floor 0.5
done
echo "ALLDONE_FLOOR_SMOKE"
