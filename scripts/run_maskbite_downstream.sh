#!/usr/bin/env bash
# MEASURE (light, verdict-relevant): does base CoBALT (balanced mask) SEPARATE from the strongest
# matched baseline (wanda-awq, wanda-sinq) on downstream as sparsity rises toward the stress boundary,
# at STILL-VIABLE 3-bit? Not the verdict grid; informs the mask proposal. SparseGPT added as the
# matched-OBS control (default percdamp under-tunes it -> also run pd=0.1).
# Usage: bash run_maskbite_downstream.sh "<sp list>" "<model list>"
set -u
cd /workspace/PTQResearch
source env.sh >/dev/null 2>&1
SPS="${1:-0.6 0.7}"
MODELS="${2:-tinyllama qwen-1.5b stablelm-2}"
BITS=3
LIMIT=1000
OUT=results/maskbite/ds
mkdir -p "$OUT"
CSV="$OUT/results.csv"
PY=$VENV/bin/python
for M in $MODELS; do
  for SP in $SPS; do
    for METHOD in cobalt wanda-awq wanda-sinq sparsegpt; do
      EXTRA=""
      [ "$METHOD" = "sparsegpt" ] && EXTRA="--sgpt-percdamp 0.1"
      $PY src/camera_bench.py --model "$M" --method "$METHOD" --sparsity "$SP" --bits "$BITS" \
          --force-true-bits --limit "$LIMIT" --csv "$CSV" \
          --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 $EXTRA \
          > "$OUT/${M}_${METHOD}_sp${SP}.log" 2>&1
      echo "[done] $M $METHOD sp$SP"
    done
  done
done
echo "ALLDONE_MASKBITE_DS"
