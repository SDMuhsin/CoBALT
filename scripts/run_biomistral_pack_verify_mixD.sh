#!/bin/bash
# Pack the mixD medical artifact (BLK1632_4 on q/k/v/o/gate|up, DENSE4 down_proj, fused qkv, DENSE4 embed/lm_head) and run the
# G6 oracle verify (no-MMA build, shipped BLOCKS/KPB/KUNROLL/PVB knobs).  usage: $0 <MIG> [raw_tag]
set -u
MIG=${1:?mig}; RAW=${2:-cobalt_mixD_b4_gptq_aw_medcal}
ROOT=/workspace/PTQResearch; cd $ROOT
source scripts/cobaltkernel_env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$MIG PYTHONUNBUFFERED=1
PY=/scratch/root/PTQResearch/env/bin/python
B=/scratch/root/PTQResearch/accel4bit_models/biomistral-7b; HF=$B/hf_bf16
OUT=$B/${RAW}_cbk1f
stamp() { echo "[$(date '+%F %T')] $*"; }
if [ ! -f $OUT/manifest.json ]; then
  stamp "pack $RAW -> $OUT"
  $PY -u src/cobaltkernel/pack_cobalt.py --raw $B/$RAW --out $OUT --layout BLK1632_4 --fuse-qkv --model-path $HF \
      --embed-layout DENSE4 --layout-override down_proj=DENSE4 2>&1 | tail -8
fi
stamp "verify (no MMA; BLOCKS 160 KPB 128 KUNROLL 4 PVB 2; PF_BLK1632_D=0)"
mkdir -p results/biomistral/verify
env -u COBALT_BLK1632_MMA COBALT_BLK1632=4 COBALT_BLK1632_D=0 COBALT_BLOCKS=160 COBALT_KPB=128 COBALT_ATTN_KUNROLL=4 COBALT_PVB=2 \
  $PY -u src/cobaltkernel/verify_kernel.py --model $OUT --config $HF --ref-packed --out results/biomistral/verify/biomistral_${RAW}_verify.txt 2>&1 | tail -25
stamp "PACK_VERIFY_DONE"
