#!/bin/bash
# CoBALT quantize BioMistral-7B with the shipped 27B recipe (REQUIREMENTS G2) -> raw artifact.
# usage: scripts/run_biomistral_quant.sh <tag> <MIG-uuid> [extra quantize_cobalt.py args...]
set -u
TAG=${1:?tag}; MIG=${2:?mig}; shift 2
source /workspace/PTQResearch/env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=$MIG PYTHONUNBUFFERED=1
HF=/scratch/root/PTQResearch/accel4bit_models/biomistral-7b/hf_bf16
OUT=/scratch/root/PTQResearch/accel4bit_models/biomistral-7b/$TAG
cd /workspace/PTQResearch
/scratch/root/PTQResearch/env/bin/python -u src/cobaltkernel/quantize_cobalt.py --model-path $HF --out $OUT \
  --sparsity 0.5 --bits 4 --beta 0.5 --group-size 128 --hull survivor --damping-frac 0.01 \
  --calib ultrachat --n-calib 128 --seq-len 2048 "$@"
echo "QUANT_DONE rc=$? tag=$TAG"
