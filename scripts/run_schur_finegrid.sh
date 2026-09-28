#!/usr/bin/env bash
# Grid-resolution check: the location result is an argmin over an 11-point beta grid, and the
# document proves Phi is a step function with finitely many breakpoints. This re-measures one
# layer on a 101-point grid (step 0.01) to test whether the coarse grid is hiding anything.
set -u
cd /workspace/PTQResearch
source env.sh
rm -f results/spectral_v2/schur_finegrid.csv
BETAS=$(python -c "print(' '.join(f'{i/100:.2f}' for i in range(101)))")
PYTHONUNBUFFERED=1 python -u src/probe_schur.py --model gemma-2b --sparsity 0.5 0.7 \
  --layers 9 --betas $BETAS --csv results/spectral_v2/schur_finegrid.csv
echo SCRIPT_DONE
