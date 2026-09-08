#!/usr/bin/env bash
# NOVEL test (#37): is the RTN-tuned beta=0.5 optimal under SINQ? Sweep cobalt-sinq beta {0.5,0.75,1.0} on
# opt/tiny/qwen @sp0.6. Under SINQ (less collapse-prone than RTN) STRONGER balance may help (on RTN beta>0.5
# collapsed). If higher beta wins >=3 => quantizer-adaptive balance = a novel finding.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/sinq_beta; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in opt-1.3b tinyllama qwen-1.5b; do
  for B in 0.75 1.0; do
    $PY src/camera_bench.py --model "$M" --method cobalt-sinq --sparsity 0.6 --bits 3 --force-true-bits \
        --cobalt-beta "$B" --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        --hp "b$B" > "$OUT/${M}_b${B}.log" 2>&1; echo "[done] $M b$B"
  done
done
echo "ALLDONE_SINQ_BETA"
