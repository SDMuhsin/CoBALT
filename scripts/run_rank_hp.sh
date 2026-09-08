#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/rankhp
for MODEL in tinyllama qwen-1.5b; do
  for SP in 0.5 0.6; do
    for COMB in magtie min; do
      C="--model $MODEL --norm col --target-global $SP --mask balanced_rank --rank-combine $COMB --col-balance-exp 0.5 --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
      echo "=== $MODEL sp$SP $COMB b0.5 ==="; python src/nosink.py $C --technique-tag ${MODEL}-$COMB-$SP 2>&1 | grep -E "RESULT.*ppl" || echo FAIL
    done
  done
done
echo "=== RANK HP DONE ==="
