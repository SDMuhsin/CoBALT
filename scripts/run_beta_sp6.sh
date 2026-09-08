#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/beta_sp6
for MODEL in tinyllama qwen-1.5b; do
  for B in 0.25 0.5 0.75 1.0 1.5; do
    C="--model $MODEL --norm col --target-global 0.6 --mask balanced --col-balance-exp $B --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
    echo "=== $MODEL sp0.6 beta$B ==="; python src/nosink.py $C --technique-tag ${MODEL}-b$B 2>&1 | grep -E "RESULT.*ppl" || echo FAIL
  done
done
echo "=== BETA SP6 DONE ==="
