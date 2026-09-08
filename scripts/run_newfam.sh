#!/usr/bin/env bash
# MEASURE-FIRST (new direction): does base cobalt beat baselines at sp0.6/3-bit on 2 MORE healthy families
# (pythia-1.4b = KNOWN CoBALT-rescue-FAILS per cobalt-win-is-bottleneck-contingent; opt-1.3b)? A family
# where CoBALT's column-balance LOSES = a DIFFERENT failure mode => room for a NEW-mechanism mask.
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
OUT=results/maskbite/newfam; mkdir -p "$OUT"; CSV="$OUT/results.csv"; PY=$VENV/bin/python
for M in pythia-1.4b opt-1.3b; do
  for MTH in cobalt wanda-sinq wanda-awq; do
    $PY src/camera_bench.py --model "$M" --method "$MTH" --sparsity 0.6 --bits 3 --force-true-bits \
        --cobalt-beta 0.5 --limit 1000 --csv "$CSV" --ds-tasks arc_easy,piqa --ppl-tasks wikitext2 \
        > "$OUT/${M}_${MTH}.log" 2>&1; echo "[done] $M $MTH"
  done
done
echo "ALLDONE_NEWFAM"
