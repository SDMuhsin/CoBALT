#!/usr/bin/env bash
# Primary measurement for the spectral_v2 document (calibration window 0-16).
# Writes results/spectral_v2/schur.csv, the artifact the measured section is generated from.
# --svd-check adds the exact ||Atilde||_2 (largest singular value) alongside the power
# iteration; the measured section uses the exact value and reports the comparison.
set -u
cd /workspace/PTQResearch
source env.sh
rm -f results/spectral_v2/schur.csv
# start from a clean file: probe_schur.py APPENDS, so a stale CSV would mix schemas
PYTHONUNBUFFERED=1 python -u src/probe_schur.py --model gemma-2b --sparsity 0.5 0.7 --svd-check \
  --csv results/spectral_v2/schur.csv
echo SCRIPT_DONE
