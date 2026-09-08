#!/usr/bin/env bash
# NOVEL (#40): adaptive-beta with CONSERVATIVE ceiling (beta_hi=0.85 via --cobalt-protect) so noisy per-matrix
# demotion never pushes a matrix to the beta=1.0 collapse. Should keep tiny's win + protect qwen. opt/tiny/qwen.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/sinq_adapt2; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in opt-1.3b tinyllama qwen-1.5b; do
  $PY src/camera_bench.py --model "$M" --method cobalt-sinq --sparsity 0.6 --bits 3 --force-true-bits \
      --cobalt-adapt 3.0 --cobalt-protect 0.85 --limit 1000 --csv "$CSV" \
      --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 --hp "acap85" > "$OUT/${M}.log" 2>&1; echo "[done] $M"
done
echo "ALLDONE_ADAPT2"
