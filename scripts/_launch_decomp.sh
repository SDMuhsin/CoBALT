#!/usr/bin/env bash
set -u
cd /workspace/PTQResearch
# Wait for the running anchor (real_prism) job to release the MIG slice.
while kill -0 640230 2>/dev/null; do sleep 20; done
echo "[launcher] anchor job done; starting 2x2 campaign $(date -u +%H:%M:%S)"
bash scripts/run_mask_norm_decomp.sh
