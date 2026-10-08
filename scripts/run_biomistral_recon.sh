#!/bin/bash
# Block-wise reconstruction of a finished CoBALT raw artifact (zero-byte, same kernel layout) -> new raw artifact
# -> fakequant export -> lm_eval, all on ONE slice.
# usage: scripts/run_biomistral_recon.sh <init_raw_tag> <out_raw_tag> <MIG-uuid> [cobalt_recon.py args...]
#   quality tag = <out_raw_tag>_e4 ; log results/biomistral/qgap/<out_raw_tag>_e4.log
set -u
RAW=${1:?init raw tag}; OUT=${2:?out raw tag}; MIG=${3:?mig}; shift 3
ROOT=/workspace/PTQResearch; cd $ROOT
source $ROOT/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$MIG PYTHONUNBUFFERED=1
PY=/scratch/root/PTQResearch/env/bin/python
B=/scratch/root/PTQResearch/accel4bit_models/biomistral-7b
TAG=${OUT}_e4; LOG=$ROOT/results/biomistral/qgap/$TAG.log
echo "[$(date '+%F %T')] RECON init=$RAW out=$OUT mig=$MIG args=$*" | tee -a $LOG
if [ ! -f $B/$OUT/manifest.json ] || ! $PY -c "import json,sys; m=json.load(open('$B/$OUT/manifest.json')); sys.exit(0 if m.get('recon',{}).get('layers_done',0)==32 else 1)"; then
  $PY -u src/cobaltkernel/cobalt_recon.py --model-path $B/hf_bf16 --art $B/$RAW --out $B/$OUT "$@" >> $LOG 2>&1
  echo "[$(date '+%F %T')] RECON_DONE rc=$?" | tee -a $LOG
fi
cp $B/$OUT/manifest.json $ROOT/results/biomistral/qgap/$TAG.manifest.json
LOG_SAMPLES=1 bash scripts/run_biomistral_fq_eval.sh $OUT $TAG $MIG >> $LOG 2>&1
echo "[$(date '+%F %T')] ARM_DONE $TAG" | tee -a $LOG
