#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/lev
for MODEL in tinyllama qwen-1.5b; do
  C="--model $MODEL --norm col --target-global 0.5 --col-balance-exp 0.5 --mask balanced_lev --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
  echo "=== $MODEL balanced_lev ==="; python src/nosink.py $C --technique-tag ${MODEL}-lev 2>&1 | grep -E "RESULT.*ppl" || echo "ARM FAILED (see below)"
done
echo "=== LEV SMOKE DONE ==="; cat $CSVDIR/*.csv 2>/dev/null | grep -v "^timestamp"
