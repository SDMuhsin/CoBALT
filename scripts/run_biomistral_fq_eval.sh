#!/bin/bash
# export fakequant (embed/lm_head DENSE4 by default) from a raw artifact, then lm_eval it under vLLM on ONE slice.
# usage: $0 <raw_tag> <quality_tag> <MIG>
# env:   EMBED_BITS (default 4), LMHEAD_BITS (default = EMBED_BITS); the fakequant dir suffix encodes them
#        (_e4 for the default, i.e. unchanged names for every existing arm; _e<E>h<H> otherwise)
set -u
RAW=${1:?raw tag}; TAG=${2:?quality tag}; MIG=${3:?mig}
EMBED_BITS=${EMBED_BITS:-4}; LMHEAD_BITS=${LMHEAD_BITS:-$EMBED_BITS}
source /workspace/PTQResearch/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$MIG PYTHONUNBUFFERED=1
PY=/scratch/root/PTQResearch/env/bin/python; ROOT=/workspace/PTQResearch; cd $ROOT
B=/scratch/root/PTQResearch/accel4bit_models/biomistral-7b
if [ "$EMBED_BITS" = 4 ] && [ "$LMHEAD_BITS" = 4 ]; then SUF=_e4; else SUF=_e${EMBED_BITS}h${LMHEAD_BITS}; fi
FQ=$B/${RAW}_fakequant_hf$SUF
[ -f $FQ/model.safetensors.index.json ] || $PY -u src/cobaltkernel/export_fakequant.py --src $B/hf_bf16 --art $B/$RAW --out $FQ \
    --embed-bits $EMBED_BITS --lm-head-bits $LMHEAD_BITS | tail -3
bash scripts/run_cobaltkernel_quality.sh biomistral-7b $FQ $MIG $TAG
echo "FQ_EVAL_DONE $TAG"
