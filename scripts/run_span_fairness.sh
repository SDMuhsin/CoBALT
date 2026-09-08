#!/usr/bin/env bash
# FAIRNESS SCREEN for spanning (the awclip lesson): does the spanning weight help the NON-column-balanced
# mask as much as it helps CoBALT? Isolation at sp0.65, 3 models:
#   cobalt b0.0     = wanda per-row + OBS + RTN (NO spanning, NO col-balance)   [reference]
#   wanda-span      = wanda per-row + SPANNING + OBS + RTN                       [+spanning on wanda]
# Compare Delta_span(wanda) = wanda-span - cobalt_b0  vs  Delta_span(cobalt) = cobalt-span - cobalt_b0.5
# (latter from spband/). If Delta_span(wanda) ~ Delta_span(cobalt) => spanning is UNIVERSAL, not CoBALT-
# specific => FAIL. Also gives spanning to the mask baselines fairly.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/span_fair; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in tinyllama qwen-1.5b stablelm-2; do
  $PY src/camera_bench.py --model "$M" --method cobalt --sparsity 0.65 --bits 3 --force-true-bits \
      --cobalt-beta 0.0 --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
      --hp "b0" > "$OUT/${M}_cobalt_b0.log" 2>&1; echo "[done] $M cobalt b0"
  $PY src/camera_bench.py --model "$M" --method wanda-span --sparsity 0.65 --bits 3 --force-true-bits \
      --cobalt-beta 0.5 --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
      --hp "wspan" > "$OUT/${M}_wandaspan.log" 2>&1; echo "[done] $M wanda-span"
done
echo "ALLDONE_SPAN_FAIR"
