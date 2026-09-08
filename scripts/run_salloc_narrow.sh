#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/narrow
C="--model tinyllama --norm col --target-global 0.5 --mask balanced --col-balance-exp 0.5 --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
echo "=== narrow [0.4,0.6] ==="; python src/nosink.py $C --sens-alloc --sp-min 0.4 --sp-max 0.6 --technique-tag cob-sa-46 2>&1 | grep -E "RESULT|sens_alloc"
echo "=== narrow [0.45,0.55] ==="; python src/nosink.py $C --sens-alloc --sp-min 0.45 --sp-max 0.55 --technique-tag cob-sa-4555 2>&1 | grep -E "RESULT|sens_alloc"
echo "=== INVERTED [0.4,0.6] (prune HIGH-sens more; sanity of direction) ==="; python src/nosink.py $C --sens-alloc --sp-min 0.4 --sp-max 0.6 --sens-alloc-file results/sens_alloc/sm_tinyllama_inv.pt --technique-tag cob-sa-inv 2>&1 | grep -E "RESULT|sens_alloc"
echo "=== NARROW DONE ==="; cat $CSVDIR/*.csv 2>/dev/null
