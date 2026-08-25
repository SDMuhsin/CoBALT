#!/bin/bash
# Downstream comparison campaign: VALOR vs PRISM vs Wanda+AWQ vs Wanda+SINQ
# 4 methods x 8 sparsities {0.0..0.7} x tasks, 3-bit, gemma-2b, full splits.
# Cheap discriminating tasks (arc_e,arc_c,hellaswag) FIRST across the whole grid,
# then full-14k MMLU. Resumable via a done-ledger. Serial on the one 2g MIG slice.
set -u
cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env.sh >/dev/null 2>&1
PY=/scratch/root/PTQResearch/env/bin/python
GRID="${GRID_DIR:-/workspace/PTQResearch/results/downstream_grid}"
LOGD=$GRID/logs
mkdir -p "$GRID" "$LOGD"
DONE=$GRID/done.log ; touch "$DONE"
MAIN=$GRID/campaign.log

SPARS="${SPARS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7}"
METHODS="${METHODS:-valor prism wanda-awq wanda-sinq}"
PASSES="${PASSES:-cheap mmlu}"   # NB: do NOT name this GROUPS (reserved bash builtin)
CHEAP_TASKS="arc_easy,arc_challenge,hellaswag"
MMLU_TASKS="mmlu"
LIMIT_FLAG="${LIMIT_FLAG:-}"   # e.g. "--downstream-limit 8" for a smoke test

is_done ()   { grep -qxF "$1" "$DONE"; }
mark_done () { echo "$1" >> "$DONE"; }

run_valor () { # $1=sparsity $2=tasks
  $PY src/nosink.py --mode vdense --hold-global --norm col --target-global "$1" --ntest 20 \
      --downstream --downstream-tasks "$2" --downstream-csv-dir "$GRID" \
      --technique-tag valor $LIMIT_FLAG
}
run_base () { # $1=technique $2=sparsity $3=tasks
  $PY benchmarks/benchmark_suite.py --model gemma-2b --technique "$1" --precision 3 \
      --sparsity "$2" --dataset wikitext2 --downstream --downstream-tasks "$3" \
      --downstream-csv-dir "$GRID" --no-csv --quiet $LIMIT_FLAG
}

for GRP in $PASSES; do
  if [ "$GRP" = "cheap" ]; then TASKS=$CHEAP_TASKS; else TASKS=$MMLU_TASKS; fi
  echo "[pass $GRP] tasks=$TASKS" >> "$MAIN"
  for SP in $SPARS; do
    for M in $METHODS; do
      KEY="$M sp$SP $GRP"
      if is_done "$KEY"; then echo "SKIP $KEY" >> "$MAIN"; continue; fi
      echo "===== START $KEY $(date +%F_%H:%M:%S) =====" >> "$MAIN"
      LOG=$LOGD/${M}_sp${SP}_${GRP}.log
      if [ "$M" = "valor" ]; then run_valor "$SP" "$TASKS" >> "$LOG" 2>&1
      else                        run_base "$M" "$SP" "$TASKS" >> "$LOG" 2>&1 ; fi
      RC=$?
      echo "===== END   $KEY rc=$RC $(date +%F_%H:%M:%S) =====" >> "$MAIN"
      [ $RC -eq 0 ] && mark_done "$KEY"
    done
  done
done
echo "CAMPAIGN DONE $(date +%F_%H:%M:%S)" >> "$MAIN"
