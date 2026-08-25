#!/bin/bash
# Phase 1 of the bias-correction ablation: qwen-0.5b PPL-only grid on wikitext2 then C4,
# full 12-technique study list x {3,4,5}bit x {5,25,50}%. Fast decisive signal (no downstream).
# Detached launch:
#   cd /workspace/PRISM
#   setsid nohup bash scripts/run_qwen_ppl_phase1.sh >> logs/abl_phase1.log 2>&1 </dev/null &
set -u
cd /workspace/PRISM
for DS in wikitext2 c4; do
    echo "[phase1] ===== dataset=$DS $(date) ====="
    ABL_MODEL=qwen-0.5b ABL_DATASET="$DS" MAXROUNDS=40 \
        bash scripts/supervise_ablation.sh
done
echo "[phase1] ALL DONE $(date)"
