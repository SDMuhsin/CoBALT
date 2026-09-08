#!/usr/bin/env bash
# SMOKE (#42): spectral-subspace-preserving mask (Jaccard 0.30 = genuinely distinct survivor set). Pure
# spectral + magnitude-blend, vs base cobalt @sp0.6/3bit, 3 healthy families. First non-perturbative candidate.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/spectral_smoke; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in tinyllama qwen-1.5b stablelm-2; do
  for MTH in cobalt-spectral cobalt-specblend cobalt; do
    $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity 0.6 --bits 3 --force-true-bits \
        --cobalt-beta 0.5 --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        > "$OUT/${M}_${MTH}.log" 2>&1; echo "[done] $M $MTH"
  done
done
echo "ALLDONE_SPECTRAL"
