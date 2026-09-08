#!/usr/bin/env bash
# SMOKE (#25 balanced_span): softer balance + spanning vs base cobalt, sp0.6/3-bit, 3 healthy models,
# with attribution (dose-only vs spanning-only vs full). arc_easy+piqa.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/span_smoke; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
run(){ local M=$1 MTH=$2 B=$3
  $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity 0.6 --bits 3 --force-true-bits \
      --cobalt-beta "$B" --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
      --hp "b$B" > "$OUT/${M}_${MTH}_b${B}.log" 2>&1; echo "[done] $M $MTH b$B"; }
for M in tinyllama qwen-1.5b stablelm-2; do
  run "$M" cobalt 0.5          # base
  run "$M" cobalt 0.4          # dose-only
  run "$M" cobalt-span 0.5     # spanning-only
  run "$M" cobalt-span 0.4     # full candidate
done
echo "ALLDONE_SPAN_SMOKE"
