#!/usr/bin/env bash
# Paper-protocol (tuned, matched, full splits, 4 reasoning tasks, 3b/4b x sp0.4-0.8) decoder suite for the
# QUANTIZER-STAGE UPGRADE: cobalt-sinq (balanced mask + OBS + SINQ dual-normalized quantizer, beta tuned)
# vs tuned matched baselines, on additional decoder families. cobalt (group-RTN, the submitted variant) runs
# LAST as the "before" arm. Models run sequentially; each dispatcher fans over every free MIG slice.
# usage: scripts/run_decoder_suite.sh <model> [<model> ...]
set -u; cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=""; source env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for M in "$@"; do
  OUT=results/decoder_suite/$M; mkdir -p "$OUT/logs"
  $VENV/bin/python scripts/camera_dispatch.py --model "$M" --tuned --force-true-bits --cobalt-group-size 128 \
      --methods cobalt-sinq,wanda-sinq,wanda-awq,sparsegpt,fp16,awq,sinq,cobalt \
      --ds-tasks arc_easy,piqa,hellaswag,winogrande --ppl-tasks "" \
      --csv "$OUT/results.csv" --log-dir "$OUT/logs" > "$OUT/dispatch.log" 2>&1
  echo "MODEL_DONE $M"
done
echo "ALLDONE_DECODER_SUITE"
