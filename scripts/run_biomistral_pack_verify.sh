#!/bin/bash
# Pack a BioMistral raw artifact into the kernel format and run the G6 oracle verify (no-MMA build, shipped knobs).
# usage: $0 <MIG> <raw_tag> [pack layout-override, e.g. down_proj=DENSE4]   env PF_D=0 to set COBALT_BLK1632_D=0 for a dense down_proj
set -u
MIG=${1:?mig}; RAW=${2:?raw tag}; OVR=${3:-}
ROOT=/workspace/PTQResearch; cd $ROOT
source scripts/cobaltkernel_env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$MIG PYTHONUNBUFFERED=1
PY=/scratch/root/PTQResearch/env/bin/python
B=/scratch/root/PTQResearch/accel4bit_models/biomistral-7b; HF=$B/hf_bf16
OUT=$B/${RAW}_cbk1f
stamp() { echo "[$(date '+%F %T')] $*"; }
if [ ! -f $OUT/manifest.json ]; then
  stamp "pack $RAW -> $OUT (override '${OVR}')"
  $PY -u src/cobaltkernel/pack_cobalt.py --raw $B/$RAW --out $OUT --layout BLK1632_4 --fuse-qkv --model-path $HF \
      --embed-layout DENSE4 ${OVR:+--layout-override $OVR} 2>&1 | tail -8
fi
stamp "verify (no MMA; BLOCKS 160 KPB 128 KUNROLL 4 PVB 2${PF_D:+; PF_BLK1632_D=$PF_D})"
mkdir -p results/biomistral/verify
env -u COBALT_BLK1632_MMA COBALT_BLK1632=4 ${PF_D:+COBALT_BLK1632_D=$PF_D} COBALT_BLOCKS=160 COBALT_KPB=128 COBALT_ATTN_KUNROLL=4 COBALT_PVB=2 \
  $PY -u src/cobaltkernel/verify_kernel.py --model $OUT --config $HF --ref-packed --out results/biomistral/verify/biomistral_${RAW}_verify.txt 2>&1 | tail -25
stamp "PACK_VERIFY_DONE"
