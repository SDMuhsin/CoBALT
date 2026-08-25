#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch
while kill -0 679400 2>/dev/null; do sleep 20; done   # wait for dual batch (K/L) to finish
echo "[chain] dual batch done; starting act-exp $(date -u +%H:%M:%S)"
bash scripts/run_act_exp.sh
