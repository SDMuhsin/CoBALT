#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/span_fair_sp6; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in tinyllama qwen-1.5b stablelm-2; do
  $PY src/camera_bench.py --model "$M" --method wanda-span --sparsity 0.6 --bits 3 --force-true-bits \
      --cobalt-beta 0.5 --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
      --hp wspan > "$OUT/${M}_wandaspan.log" 2>&1; echo "[done] $M wanda-span sp0.6"
done
echo "ALLDONE_SPANFAIR_SP6"
