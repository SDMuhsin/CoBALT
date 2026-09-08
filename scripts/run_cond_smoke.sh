#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/cond_smoke; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in tinyllama qwen-1.5b stablelm-2; do
  $PY src/camera_bench.py --model "$M" --method cobalt-cond --sparsity 0.6 --bits 3 --force-true-bits \
      --cobalt-beta 0.5 --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
      > "$OUT/${M}_cond.log" 2>&1; echo "[done] $M cobalt-cond"
done
echo "ALLDONE_COND"
