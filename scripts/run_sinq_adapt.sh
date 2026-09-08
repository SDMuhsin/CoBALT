#!/usr/bin/env bash
# NOVEL (#39): ADAPTIVE per-matrix beta from the demotion-rate signal (opt=low->strong balance, qwen=high->
# moderate). Should capture opt's strong-balance gain WITHOUT collapsing tiny/qwen -> beat fixed beta on >=3.
# slope {3,5}. opt/tiny/qwen @sp0.6 under SINQ. Compare: b0.5 opt.405/tiny.385/qwen.480; b0.75 .417/.393/.493
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/sinq_adapt; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in opt-1.3b tinyllama qwen-1.5b; do
  for A in 3.0 5.0; do
    $PY src/camera_bench.py --model "$M" --method cobalt-sinq --sparsity 0.6 --bits 3 --force-true-bits \
        --cobalt-adapt "$A" --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        --hp "adapt$A" > "$OUT/${M}_a${A}.log" 2>&1; echo "[done] $M adapt$A"
  done
done
echo "ALLDONE_SINQ_ADAPT"
