#!/usr/bin/env bash
# Multi-passage robustness for the spectral_v2 measurement. Each --take-slice k k+1 draw is
# ONE 512-token calibration sequence, of which the production routine keeps the first 256
# rows -- so each of these is a distinct 256-token passage, and together they answer whether
# the headline statistics are an artefact of any single passage.
set -u
cd /workspace/PTQResearch
source env.sh
for K in 4 8 20 40 60; do
  rm -f "results/spectral_v2/schur_draw_s${K}.csv"
  PYTHONUNBUFFERED=1 python -u src/probe_schur.py --model gemma-2b --sparsity 0.5 0.7 \
    --take-slice "$K" "$((K+1))" --draw "S$K" \
    --csv "results/spectral_v2/schur_draw_s${K}.csv"
done
echo SCRIPT_DONE
