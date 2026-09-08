#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/novel_sp6
for MODEL in tinyllama qwen-1.5b; do
  for MASK in balanced balanced_fint balanced_lev; do
    C="--model $MODEL --norm col --target-global 0.6 --col-balance-exp 0.5 --mask $MASK --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
    echo "=== $MODEL sp0.6 $MASK ==="; python src/nosink.py $C --technique-tag ${MODEL}-${MASK}-6 2>&1 | grep -E "RESULT.*ppl" || echo FAIL
  done
done
echo "=== NOVEL SP6 DONE ==="
