#!/usr/bin/env bash
# SMOKE (#30): per-head balanced o_proj (cobalt-perhead) vs base cobalt @sp0.6/3-bit, 3 healthy models.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/perhead_smoke; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in tinyllama qwen-1.5b stablelm-2; do
  for MTH in cobalt-perhead cobalt; do
    $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity 0.6 --bits 3 --force-true-bits \
        --cobalt-beta 0.5 --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        > "$OUT/${M}_${MTH}.log" 2>&1; echo "[done] $M $MTH"
  done
done
echo "ALLDONE_PERHEAD"
