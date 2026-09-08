#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/rankmagw
run(){ local M=$1 SP=$2 W=$3; local C="--model $M --norm col --target-global $SP --mask balanced_rank --rank-combine magtie --rank-magw $W --col-balance-exp 0.5 --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
  echo "=== $M sp$SP magw$W ==="; python src/nosink.py $C --technique-tag $M-w$W-$SP 2>&1 | grep -E "RESULT.*ppl" || echo FAIL; }
for W in 0.2 0.5 1.0; do run qwen-1.5b 0.5 $W; run qwen-1.5b 0.6 $W; done
for W in 0.5 1.0; do run tinyllama 0.6 $W; done
echo "=== MAGW DONE ==="
