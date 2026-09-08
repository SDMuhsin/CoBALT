#!/usr/bin/env bash
# NOVEL (#37): STRONG balance (beta=1.0) + COLUMN FLOOR under SINQ. Hypothesis: floor prevents the beta=1.0
# starvation-collapse on tiny/qwen while keeping opt's big beta=1.0 gain (+3.5). If beta1.0+floor beats
# beta=0.75(best fixed) on >=3 => novel adaptive-equivalent mask. opt/tiny/qwen @sp0.6.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/sinq_bfloor; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in opt-1.3b tinyllama qwen-1.5b; do
  for FF in 0.25 0.5; do
    $PY src/camera_bench.py --model "$M" --method cobalt-sinq --sparsity 0.6 --bits 3 --force-true-bits \
        --cobalt-beta 1.0 --cobalt-floor-frac "$FF" --limit 1000 --csv "$CSV" \
        --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 --hp "b1.0f$FF" > "$OUT/${M}_f${FF}.log" 2>&1
    echo "[done] $M b1.0 floor$FF"
  done
done
echo "ALLDONE_SINQ_BFLOOR"
