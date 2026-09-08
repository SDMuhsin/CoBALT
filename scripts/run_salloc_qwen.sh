#!/usr/bin/env bash
set -u; cd /workspace/PTQResearch; source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"; LIM=1000; CSVDIR=results/sens_alloc/qwen
C="--model qwen-1.5b --norm col --target-global 0.5 --mask balanced --col-balance-exp 0.5 --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"
echo "=== qwen uniform ==="; python src/nosink.py $C --technique-tag q-unif 2>&1 | grep -E "RESULT|sens_alloc"
echo "=== qwen salloc [0.45,0.55] ==="; python src/nosink.py $C --sens-alloc --sp-min 0.45 --sp-max 0.55 --technique-tag q-sa-4555 2>&1 | grep -E "RESULT|sens_alloc"
echo "=== qwen salloc [0.4,0.6] ==="; python src/nosink.py $C --sens-alloc --sp-min 0.4 --sp-max 0.6 --technique-tag q-sa-46 2>&1 | grep -E "RESULT|sens_alloc"
echo "=== QWEN DONE ==="; cat $CSVDIR/*.csv 2>/dev/null | grep -v "^timestamp"
