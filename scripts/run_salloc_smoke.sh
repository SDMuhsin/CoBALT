#!/usr/bin/env bash
# SMOKE (one cell): global-sensitivity SPARSITY ALLOCATION mask, tinyllama (HEALTHY regime), 3-bit sp0.5.
# 4 arms to answer BOTH questions at once:
#   (1) does allocation help CoBALT downstream?      cob-salloc vs cob-unif
#   (2) is allocation a UNIVERSAL lever (fairness)?  wan-salloc vs wan-unif  (does wanda gain the same?)
# All arms: 3-bit, group-64, global sparsity 0.5 (matched bpw: bitmap/scale/zero identical), norm=col.
set -u
cd /workspace/PTQResearch
source env.sh >/dev/null 2>&1
DS="arc_easy,piqa"
LIM=1000
CSVDIR=results/sens_alloc/smoke
COMMON="--model tinyllama --norm col --target-global 0.5 --downstream --downstream-tasks $DS --downstream-limit $LIM --downstream-csv-dir $CSVDIR"

GF() { grep -E "RESULT|DOWNSTREAM|sens_alloc|\bacc\b|accuracy|arc_easy|piqa|Task |: 0\.|=0\." ; }
echo "===== ARM 1/4: cobalt-uniform ====="
python src/nosink.py $COMMON --mask balanced --col-balance-exp 0.5 --technique-tag cob-unif 2>&1 | GF
echo "===== ARM 2/4: cobalt-salloc ====="
python src/nosink.py $COMMON --mask balanced --col-balance-exp 0.5 --sens-alloc --technique-tag cob-salloc 2>&1 | GF
echo "===== ARM 3/4: wanda-uniform ====="
python src/nosink.py $COMMON --mask wanda --technique-tag wan-unif 2>&1 | GF
echo "===== ARM 4/4: wanda-salloc ====="
python src/nosink.py $COMMON --mask wanda --sens-alloc --technique-tag wan-salloc 2>&1 | GF
echo "===== SMOKE DONE ====="
echo "--- CSV dump ---"; cat $CSVDIR/*.csv 2>/dev/null
