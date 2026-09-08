#!/bin/bash
# Real DOWNSTREAM grid: cobalt-awclip vs baseline suite (with AND without awclip), matched bpw.
# Answers "how many downstream wins does cobalt-awclip get vs baselines" -- the deliverable never run.
# 3-bit, sp0.5, g128, --force-true-bits. sparsegpt percdamp {0.01,0.1} (don't under-tune the OBS control).
set -u
cd /workspace/PTQResearch
source env.sh >/dev/null 2>&1
PY="$VENV/bin/python"
CSV=results/awclip_downstream/results.csv
LOGD=results/awclip_downstream/logs
mkdir -p "$LOGD"
# our arm + S4 control + baselines WITHOUT awclip (the practical comparison) + FAIR arms (awclip given to baselines)
METHODS="cobalt-awclip,cobalt,wanda-awq,wanda-sinq,sparsegpt,wanda-awq-awclip,wanda-sinq-awclip"
DS="arc_easy,piqa,hellaswag,winogrande"
for M in gemma-2b tinyllama qwen-1.5b; do
  echo "===== DISPATCH $M ====="
  "$PY" scripts/camera_dispatch.py --model "$M" --methods "$METHODS" \
    --sparsities 0.5 --bits 3 --cobalt-group-size 128 --force-true-bits \
    --ds-tasks "$DS" --ppl-tasks wikitext2 \
    --sgpt-percdamps 0.01,0.1 --col-balance-exps 0.5 \
    --free-gb 14 --csv "$CSV" --log-dir "$LOGD"
done
echo "ALL_MODELS_DONE"
