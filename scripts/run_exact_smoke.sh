#!/usr/bin/env bash
# SMOKE (attempt-11 #18): EXACT doubly-balanced mask (cobalt-exact) vs base cobalt at sp0.6/3-bit (the
# confirmed healthy collapse-onset point where base cobalt already beats baselines). tiny+qwen, arc+piqa.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/exact_smoke; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in tinyllama qwen-1.5b; do
  for MTH in cobalt-exact cobalt; do
    $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity 0.6 --bits 3 --force-true-bits \
        --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        > "$OUT/${M}_${MTH}.log" 2>&1
    echo "[done] $M $MTH"
  done
done
echo "ALLDONE_EXACT_SMOKE"
