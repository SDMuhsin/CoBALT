#!/usr/bin/env bash
# CoBALT-seq (cross-layer v2) smoke with attribution: cobalt (dense inputs, fixed beta=0.5) vs
# cobalt-seqfix (compressed-prefix inputs, fixed beta=0.5) vs cobalt-seq (compressed-prefix inputs,
# per-block beta by propagated error). Paper's collapse band: gemma-2b 3-bit sp0.6/0.7 + qwen-1.5b sp0.6.
# usage: scripts/run_cobalt_seq_smoke.sh <MIG-UUID> <model> <sp> [<sp> ...]
set -u; cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=$1; M=$2; shift 2
source env.sh >/dev/null 2>&1
OUT=results/cobalt_seq/smoke; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for SP in "$@"; do
  for MTH in cobalt-seq cobalt-seqfix cobalt; do
    $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity $SP --bits 3 --force-true-bits \
        --cobalt-group-size 128 --cobalt-beta 0.5 --limit 1000 --csv "$CSV" \
        --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 > "$OUT/${M}_${MTH}_sp${SP}.log" 2>&1
    echo "[done] $M $MTH sp$SP"
  done
done
echo "ALLDONE_SEQ_SMOKE_$M"
