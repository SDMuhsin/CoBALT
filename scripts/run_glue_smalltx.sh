#!/usr/bin/env bash
# 3-bit MATCHED quant+prune grid on SMALLER / non-BERT encoder transformers, MNLI.
# One slice, models run SEQUENTIALLY (MIG contention kills concurrent jobs). Resumable.
set -u
cd "$(dirname "$0")/.."
source env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python
CSV=results/benchmark_camera_glue/glue.csv
METHODS=cobalt,sparsegpt,wanda-awq,wanda-sinq,awq,sinq,fp16
SP=0.4,0.5,0.55,0.6,0.65

MODELS=(
  "howey/electra-base-mnli"
  "M-FAC/bert-mini-finetuned-mnli"
  "typeform/distilbert-base-uncased-mnli"
  "cross-encoder/nli-distilroberta-base"
  "microsoft/deberta-base-mnli"
  "microsoft/deberta-large-mnli"
)
for M in "${MODELS[@]}"; do
  echo "=================== $M ==================="
  $PY src/camera_bench_glue.py --model "$M" --task mnli \
      --methods "$METHODS" --sparsities "$SP" --bits 3 --group-size 128 \
      --csv "$CSV" 2>&1
done
echo "ALL_DONE_GLUE_SMALLTX"
