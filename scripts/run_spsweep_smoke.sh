#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/spsweep
for MODEL in tinyllama qwen-1.5b; do
  for SP in 0.5 0.6 0.7; do
    C="--model $MODEL --norm col --target-global $SP --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
    echo "=== $MODEL sp$SP balanced ==="; python src/nosink.py $C --mask balanced --col-balance-exp 0.5 --technique-tag ${MODEL}-bal-$SP 2>&1 | grep -E "RESULT.*ppl" || echo FAIL
    echo "=== $MODEL sp$SP wanda ==="; python src/nosink.py $C --mask wanda --technique-tag ${MODEL}-wan-$SP 2>&1 | grep -E "RESULT.*ppl" || echo FAIL
  done
done
echo "=== SPSWEEP DONE ==="
