#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/robust
for MODEL in tinyllama qwen-1.5b; do
  for SP in 0.5 0.6; do
    for MASK in balanced balanced_robust; do
      C="--model $MODEL --norm col --target-global $SP --mask $MASK --col-balance-exp 0.5 --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
      echo "=== $MODEL sp$SP $MASK ==="; python src/nosink.py $C --technique-tag $MODEL-$MASK-$SP 2>&1 | grep -E "RESULT.*ppl" || echo FAIL
    done
  done
done
echo "=== ROBUST SMOKE DONE ==="
