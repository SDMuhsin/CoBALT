#!/usr/bin/env bash
# MEASURE: does reducing column-balance beta RECOVER opt/pythia (where balance collapses cobalt)? If beta=0
# (=per-row wanda+OBS) recovers them while beta=0.5 wins gemma-lineage => an ADAPTIVE-beta mask is the novel
# unified lever. cobalt beta {0.0,0.25,0.5} on opt-1.3b + pythia-1.4b @sp0.6.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/beta_recover; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in opt-1.3b pythia-1.4b; do
  for B in 0.0 0.25 0.5; do
    $PY src/camera_bench.py --model "$M" --method cobalt --sparsity 0.6 --bits 3 --force-true-bits \
        --cobalt-beta "$B" --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        --hp "b$B" > "$OUT/${M}_b${B}.log" 2>&1; echo "[done] $M b$B"
  done
done
echo "ALLDONE_BETA_RECOVER"
