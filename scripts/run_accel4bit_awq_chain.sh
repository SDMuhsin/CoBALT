#!/usr/bin/env bash
# Unattended chain for arm D: wait for the running medgemma-27b AWQ quantization, then run
# 27B gen/eval/speed/bpw, then the gemma-3-4b gen/eval/speed (all serialized on one slice).
MIG=${1:-MIG-12daede5-c316-57d7-bd83-a55c417864cd}
ROOT=/workspace/PTQResearch; R=$ROOT/results/accel4bit
S=$ROOT/scripts/run_accel4bit_awq.sh
PY=/scratch/root/PTQResearch/env_accel_awq/bin/python
echo "[$(date '+%F %T')] chain start; waiting for 27B QUANT_EXIT"
until grep -q "QUANT_EXIT" $R/medgemma-27b/awq/quant.log 2>/dev/null; do sleep 60; done
sleep 20
echo "[$(date '+%F %T')] 27B quant: $(grep QUANT_EXIT $R/medgemma-27b/awq/quant.log)"
if grep -q "QUANT_EXIT 0" $R/medgemma-27b/awq/quant.log; then
  bash $S medgemma-27b $MIG gen,eval,speed,bpw
  $PY $ROOT/src/accel4bit_awq_summarize.py $R/medgemma-27b/awq $MIG
else
  echo "[$(date '+%F %T')] 27B quant FAILED — skipping 27B evals"
fi
echo "[$(date '+%F %T')] gemma-3-4b: quality eval runs on the spare slice; 4b speed is covered by phase-3 (48 GB slice) -> skipped here"
if [ -s $R/gemma-3-4b/awq/eval_medmcqa.json ] && [ -s $R/gemma-3-4b/awq/eval_wikitext.json ]; then
  echo "[$(date '+%F %T')] 4b eval already present"
else
  bash $S gemma-3-4b $MIG gen,eval
fi
$PY $ROOT/src/accel4bit_awq_summarize.py $R/gemma-3-4b/awq $MIG
echo "[$(date '+%F %T')] CHAIN_DONE"
