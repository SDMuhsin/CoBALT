#!/usr/bin/env bash
# Verify base-cobalt's apparent sp0.6 healthy edge vs the STRONGEST matched baselines (wanda-sinq +
# tuned SparseGPT pd=0.1). If wanda-sinq/sparsegpt also ~= cobalt, wanda-awq was just weak (no base edge);
# if cobalt still wins, sp0.6 is a real healthy-regime mask-lever regime (collapse-onset rescue).
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/strongbase_sp0.6; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in tinyllama qwen-1.5b; do
  for MTH in wanda-sinq sparsegpt; do
    EXTRA=""; [ "$MTH" = "sparsegpt" ] && EXTRA="--sgpt-percdamp 0.1"
    $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity 0.6 --bits 3 --force-true-bits \
        --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 $EXTRA \
        > "$OUT/${M}_${MTH}.log" 2>&1
    echo "[done] $M $MTH"
  done
done
echo "ALLDONE_STRONGBASE_SP6"
