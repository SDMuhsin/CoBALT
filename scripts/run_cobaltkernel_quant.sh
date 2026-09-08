#!/bin/bash
# CoBALT memory-streamed layer-wise quantizer runner.
#   scripts/run_cobaltkernel_quant.sh <gemma-3-4b|medgemma-27b> [MIG-UUID] [extra args...]
# Long runs:  setsid nohup scripts/run_cobaltkernel_quant.sh medgemma-27b MIG-xxx > log 2>&1 </dev/null & disown
set -u
MODEL=${1:?model-short (gemma-3-4b|medgemma-27b)}
MIG=${2:-}
shift 2 2>/dev/null || shift $#

ROOT=/workspace/PTQResearch
ENV=/scratch/root/PTQResearch/env               # repo venv: torch 2.13 + transformers 4.55 (Gemma3 OK)
PY=$ENV/bin/python
MODELS=/scratch/root/PTQResearch/accel4bit_models

case $MODEL in
  gemma-3-4b)   SRC=$MODELS/gemma-3-4b/text_bf16 ;;
  medgemma-27b) SRC=$(ls -d /scratch/ckp908/prism_hf/hub/models--unsloth--medgemma-27b-text-it/snapshots/*/ | head -1) ;;
  *) SRC=$MODEL; MODEL=$(basename "$MODEL") ;;
esac

SP=${SPARSITY:-0.5}; B=${BITS:-4}; G=${GROUP:-128}; BETA=${BETA:-0.5}; HULL=${HULL:-survivor}
NCAL=${NCALIB:-128}; SEQ=${SEQLEN:-2048}
TAG=${TAG:-cobalt_sp${SP}_b${B}_g${G}}
OUT=${OUT:-$MODELS/$MODEL/$TAG}

unset PYTHONPATH
export PIP_CONFIG_FILE=/dev/null
export HF_HOME=/scratch/ckp908/prism_hf HF_HUB_CACHE=/scratch/ckp908/prism_hf/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/scratch/root/PTQResearch/tmp
C=/scratch/root/PTQResearch/cache; mkdir -p $C/xdg $C/nv $TMPDIR
export XDG_CACHE_HOME=$C/xdg CUDA_CACHE_PATH=$C/nv
[ -n "$MIG" ] && export CUDA_VISIBLE_DEVICES=$MIG
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"
echo "[runner] model=$MODEL src=$SRC out=$OUT mig=${CUDA_VISIBLE_DEVICES:-default}"
exec $PY -u $ROOT/src/cobaltkernel/quantize_cobalt.py \
  --model-path "$SRC" --out "$OUT" \
  --sparsity $SP --bits $B --group-size $G --beta $BETA --hull $HULL \
  --calib ultrachat --calib-file $ROOT/results/accel4bit/calib_ultrachat_512x2048.txt \
  --n-calib $NCAL --seq-len $SEQ --device cuda "$@"
