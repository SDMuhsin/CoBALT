#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/rankgen; SP=0.6
for MODEL in pythia-1.4b opt-1.3b stablelm-2; do
  B="--model $MODEL --norm col --target-global $SP --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
  echo "=== $MODEL base ==="; python src/nosink.py $B --mask balanced --col-balance-exp 0.5 --technique-tag $MODEL-base 2>&1 | grep -E "RESULT.*ppl" || echo FAIL
  echo "=== $MODEL rank ==="; python src/nosink.py $B --mask balanced_rank --rank-combine magtie --rank-magw 0.001 --col-balance-exp 0.5 --technique-tag $MODEL-rank 2>&1 | grep -E "RESULT.*ppl" || echo FAIL
done
echo "=== RANK GEN DONE ==="
