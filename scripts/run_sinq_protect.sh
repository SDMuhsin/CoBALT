#!/usr/bin/env bash
# NOVEL (#38): saliency-PROTECTED strong balance (beta=1.0 + protect top-p of budget by raw importance).
# Targets the beta=1.0 collapse (=demotion of important entries). If protected beta=1.0 keeps opt's +3.5
# AND un-collapses tiny/qwen -> beats best fixed beta=0.75 on >=3 => novel mask. opt/tiny/qwen @sp0.6.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/sinq_protect; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in opt-1.3b tinyllama qwen-1.5b; do
  for P in 0.3 0.5; do
    $PY src/camera_bench.py --model "$M" --method cobalt-sinq --sparsity 0.6 --bits 3 --force-true-bits \
        --cobalt-beta 1.0 --cobalt-protect "$P" --limit 1000 --csv "$CSV" \
        --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 --hp "b1.0p$P" > "$OUT/${M}_p${P}.log" 2>&1
    echo "[done] $M protect$P"
  done
done
echo "ALLDONE_SINQ_PROTECT"
