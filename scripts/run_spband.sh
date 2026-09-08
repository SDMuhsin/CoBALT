#!/usr/bin/env bash
# MEASURE (#27 sp-band / lever #17): does the mask lever GROW with stress? cobalt vs cobalt-span(b0.5,
# the qwen-pocket lever) vs strongest baseline wanda-sinq at sp0.65 (between viable 0.6 & near-chance 0.7),
# 3 healthy models. Combined with existing sp0.6 -> the slope. Gate: viable if base cobalt arc >> chance.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/spband; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in tinyllama qwen-1.5b stablelm-2; do
  for cfg in "cobalt 0.5" "cobalt-span 0.5" "wanda-sinq 0.5"; do
    set -- $cfg; MTH=$1; B=$2
    $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity 0.65 --bits 3 --force-true-bits \
        --cobalt-beta "$B" --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        --hp "sp65" > "$OUT/${M}_${MTH}.log" 2>&1; echo "[done] $M $MTH sp0.65"
  done
done
echo "ALLDONE_SPBAND"
