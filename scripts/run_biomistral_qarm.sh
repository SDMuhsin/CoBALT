#!/bin/bash
# Quantizer-gap arm driver: quantize (if the raw artifact is missing) -> fakequant export -> lm_eval, all on ONE slice.
# usage: scripts/run_biomistral_qarm.sh <raw_tag> <quality_tag> <MIG-uuid> [quantize_cobalt.py args...]
#   env: EMBED_BITS / LMHEAD_BITS forwarded to run_biomistral_fq_eval.sh (default 4/4)
#   log: results/biomistral/qgap/<quality_tag>.log ; manifest copied to results/biomistral/qgap/<quality_tag>.manifest.json
set -u
RAW=${1:?raw tag}; TAG=${2:?quality tag}; MIG=${3:?mig}; shift 3
ROOT=/workspace/PTQResearch; cd $ROOT
B=/scratch/root/PTQResearch/accel4bit_models/biomistral-7b
LOG=$ROOT/results/biomistral/qgap/$TAG.log
echo "[$(date '+%F %T')] ARM raw=$RAW tag=$TAG mig=$MIG embed=${EMBED_BITS:-4}/${LMHEAD_BITS:-${EMBED_BITS:-4}} quant_args=$*" | tee -a $LOG
if [ ! -f $B/$RAW/manifest.json ] || ! python3 -c "import json,sys; m=json.load(open('$B/$RAW/manifest.json')); sys.exit(0 if m.get('layers_processed',0)==32 or m['config'].get('layers_processed')==32 else 1)" 2>/dev/null; then
  bash scripts/run_biomistral_quant.sh $RAW $MIG "$@" >> $LOG 2>&1
fi
cp $B/$RAW/manifest.json $ROOT/results/biomistral/qgap/$TAG.manifest.json 2>/dev/null
grep -E "DONE|eout_ho_total" $LOG | tail -1 | tee -a $LOG >/dev/null
bash scripts/run_biomistral_fq_eval.sh $RAW $TAG $MIG >> $LOG 2>&1
echo "[$(date '+%F %T')] ARM_DONE $TAG" | tee -a $LOG
