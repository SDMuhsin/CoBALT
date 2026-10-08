#!/bin/bash
# After quantize: fakequant export (-> quality eval on a 2nd slice) + DENSE4 control quantize + pack both arms.
# usage: scripts/run_biomistral_pack_chain.sh <MIG-quant-slice> <MIG-eval-slice>
set -u
MIGQ=${1:?quant slice}; MIGE=${2:?eval slice}
source /workspace/PTQResearch/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$MIGQ PYTHONUNBUFFERED=1
PY=/scratch/root/PTQResearch/env/bin/python
ROOT=/workspace/PTQResearch; cd $ROOT
B=/scratch/root/PTQResearch/accel4bit_models/biomistral-7b
HF=$B/hf_bf16
stamp() { echo "[$(date '+%F %T')] $*"; }

stamp "export fakequant (blk32, embed-bits 4)"
$PY -u src/cobaltkernel/export_fakequant.py --src $HF --art $B/cobalt_sp0.5_b4_g128_blk32 \
    --out $B/cobalt_sp0.5_b4_g128_blk32_fakequant_hf_e4 --embed-bits 4 || { echo EXPORT_FAIL; exit 1; }
stamp "launch quality eval of the fakequant on $MIGE"
setsid nohup bash scripts/run_cobaltkernel_quality.sh biomistral-7b $B/cobalt_sp0.5_b4_g128_blk32_fakequant_hf_e4 $MIGE cobalt_blk32_b4_e4 \
    > results/biomistral/quality_blk32_b4_e4.log 2>&1 < /dev/null & disown

stamp "quantize DENSE4 control raw (global top-k, same recipe)"
bash scripts/run_biomistral_quant.sh cobalt_sp0.5_b4_g128 $MIGQ 2>&1 | grep -E "DONE|QUANT_DONE|Error|error" | tail -3

stamp "pack BLK1632_4 (+fuse-qkv, DENSE4 embed/lm_head)"
$PY -u src/cobaltkernel/pack_cobalt.py --raw $B/cobalt_sp0.5_b4_g128_blk32 --out $B/cobalt_sp0.5_b4_g128_blk32_cbk1_b1632_4f \
    --layout BLK1632_4 --fuse-qkv --model-path $HF --embed-layout DENSE4 2>&1 | tail -8
stamp "pack DENSE4 control (+fuse-qkv)"
$PY -u src/cobaltkernel/pack_cobalt.py --raw $B/cobalt_sp0.5_b4_g128 --out $B/cobalt_sp0.5_b4_g128_cbk1_dense4f \
    --layout DENSE4 --fuse-qkv --model-path $HF --embed-layout DENSE4 2>&1 | tail -8
stamp "CHAIN_DONE"
