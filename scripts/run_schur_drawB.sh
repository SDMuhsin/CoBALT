#!/usr/bin/env bash
# Replication draw for the spectral_v2 measurement: re-selects the mask on a DISJOINT
# calibration window (sequences 16-32 against the measurement's 0-16) and re-measures.
set -u
cd /workspace/PTQResearch
source env.sh
rm -f results/spectral_v2/schur_drawB.csv
# start from a clean file: probe_schur.py APPENDS, so a stale CSV would mix schemas
PYTHONUNBUFFERED=1 python -u src/probe_schur.py --model gemma-2b --sparsity 0.5 0.7 \
  --take-slice 16 32 --draw B --csv results/spectral_v2/schur_drawB.csv
echo SCRIPT_DONE
