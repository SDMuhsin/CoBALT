#!/usr/bin/env bash
# SEQFIX SUITE, slice-PINNED (2026-09-03). Every cell of one model runs on ONE MIG slice, sequentially.
#
# WHY: MEASURED on this box, the same cell on two different slices differs by 9.1% PPL and ~0.4 pt
# accuracy (gemma-2b cobalt 3b sp0.6: 57.7640/.4983/.6589 on a 1g slice vs 63.5620/.5021/.6545 on the 2g
# slice, each reproducible 3/3). cuBLAS selects different kernels/reduction orders per SM count, and a
# collapse-edge cell amplifies it. camera_dispatch.py places cells on whatever slice is free, so matched
# arms can land on different slices -- a confound LARGER than the effect being measured (seqfix task-mean
# is ~0.1-0.5 pt). Pinning a model's whole grid to one slice makes the offset common-mode across arms, so
# the ARM-TO-ARM DELTA (the only quantity claimed) is clean. Models never get compared to each other, so
# different models may sit on different slices.
#
# usage: scripts/run_seqfix_pinned.sh <MIG-UUID> <model>
set -u; cd /workspace/PTQResearch
U=$1; M=$2
export CUDA_VISIBLE_DEVICES="$U"
source env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES="$U"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/seqfix_suite/$M; mkdir -p "$OUT/logs"
CSV="$OUT/results.csv"
echo "$U" > "$OUT/SLICE.txt"
for B in 3 4; do
  for SP in 0.4 0.5 0.6 0.7 0.8; do
    # matched arms run back-to-back on the SAME pinned slice
    for MTH in cobalt cobalt-seqfix cobalt-awclip cobalt-seqfix-awclip; do
      TAG="${MTH}_b${B}_sp${SP}"
      $VENV/bin/python src/camera_bench.py --model "$M" --method "$MTH" --sparsity $SP --bits $B \
          --force-true-bits --cobalt-group-size 128 --cobalt-beta 0.5 --hp beta=0.5 \
          --csv "$CSV" --ds-tasks arc_easy,piqa,hellaswag,winogrande --ppl-tasks wikitext2 \
          >> "$OUT/logs/${TAG}.log" 2>&1
      echo "[$(date +%H:%M:%S)] done $M $TAG rc=$?"
    done
  done
done
echo "ALLDONE_PINNED_$M"
