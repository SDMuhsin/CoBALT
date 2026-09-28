#!/usr/bin/env bash
set -u
cd /workspace/PTQResearch
source env.sh
rm -f results/spectral_v2/schur_heldout.csv
# start from a clean file: probe_schur.py APPENDS, so a stale CSV would mix schemas
PYTHONUNBUFFERED=1 python -u src/probe_schur.py --model gemma-2b --sparsity 0.5 0.7 \
  --heldout-slice 32 48 --csv results/spectral_v2/schur_heldout.csv
echo SCRIPT_DONE
