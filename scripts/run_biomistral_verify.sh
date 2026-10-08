#!/bin/bash
# Kernel correctness gate for BioMistral-7B (REQUIREMENTS G6): torch oracle from the SAME packed bytes,
# >=98.5% argmax agreement, M=4 bit-exact vs four M=1 runs.  usage: $0 <MIG> [arms: b1632_4 dense4]
set -u
MIG=${1:?mig}; shift; ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(b1632_4 dense4)
ROOT=/workspace/PTQResearch; cd $ROOT
source scripts/cobaltkernel_env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$MIG PYTHONUNBUFFERED=1
PY=/scratch/root/PTQResearch/env/bin/python
B=/scratch/root/PTQResearch/accel4bit_models/biomistral-7b
OUT=results/biomistral/verify; mkdir -p $OUT
stamp() { echo "[$(date '+%F %T')] $*"; }
for arm in "${ARMS[@]}"; do
  case $arm in
    b1632_4) stamp "BLK1632_4 arm"; COBALT_BLK1632=4 $PY -u src/cobaltkernel/verify_kernel.py --model $B/cobalt_sp0.5_b4_g128_blk32_cbk1_b1632_4f --config $B/hf_bf16 --ref-packed --out $OUT/biomistral_b1632_4_verify.txt 2>&1 | tail -30 ;;
    dense4)  stamp "DENSE4 arm (KPB=256)"; env -u COBALT_BLK1632 COBALT_KPB=256 $PY -u src/cobaltkernel/verify_kernel.py --model $B/cobalt_sp0.5_b4_g128_cbk1_dense4f --config $B/hf_bf16 --ref-packed --out $OUT/biomistral_dense4_verify.txt 2>&1 | tail -30 ;;
  esac
done
stamp "VERIFY_DONE"
