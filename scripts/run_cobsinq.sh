#!/usr/bin/env bash
# SMOKE (#36): cobalt-sinq (balanced mask + SINQ) vs wanda-sinq (wanda mask + SINQ) = MASK isolated under
# matched quantizer. opt/pythia (does balance help where RTN-cobalt collapsed?) + tiny/qwen/stab (does
# balance still help under SINQ?). sp0.6/3-bit.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/cobsinq; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in opt-1.3b pythia-1.4b tinyllama qwen-1.5b stablelm-2; do
  for MTH in cobalt-sinq wanda-sinq; do
    $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity 0.6 --bits 3 --force-true-bits \
        --cobalt-beta 0.5 --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        > "$OUT/${M}_${MTH}.log" 2>&1; echo "[done] $M $MTH"
  done
done
echo "ALLDONE_COBSINQ"
