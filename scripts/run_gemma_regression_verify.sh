#!/bin/bash
# Regression gate for the model-family port: the Gemma3 arms must still pass verify_kernel
# (98.5% argmax vs the torch oracle, M=4 bit-exact) after the arch.py/lm_head/norm-bypass changes.
# usage: scripts/run_gemma_regression_verify.sh <MIG-uuid>
set -u
MIG=${1:?mig}
ROOT=/workspace/PTQResearch; cd $ROOT
source scripts/cobaltkernel_env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$MIG PYTHONUNBUFFERED=1
PY=/scratch/root/PTQResearch/env/bin/python
G=/scratch/root/PTQResearch/accel4bit_models/gemma-3-4b
OUT=results/biomistral/regress; mkdir -p $OUT
stamp() { echo "[$(date '+%F %T')] $*"; }
stamp "DENSE4 arm (COBALT_KPB=256)"
env -u COBALT_BLK1632 COBALT_KPB=256 $PY -u src/cobaltkernel/verify_kernel.py --model $G/cobalt_sp0.5_b4_g128_cbk1_dense4f --config $G/cobalt_sp0.5_b4_g128_fakequant_hf --ref-packed \
    --out $OUT/gemma-3-4b_dense4_verify.txt 2>&1 | tail -25
stamp "BLK1632_4 arm (COBALT_BLK1632=4)"
COBALT_BLK1632=4 $PY -u src/cobaltkernel/verify_kernel.py --model $G/cobalt_sp0.5_b4_g128_blk32_cbk1_b1632_4f --config $G/cobalt_sp0.5_b4_g128_fakequant_hf --ref-packed \
    --out $OUT/gemma-3-4b_b1632_4_verify.txt 2>&1 | tail -25
stamp "REGRESS_DONE"
