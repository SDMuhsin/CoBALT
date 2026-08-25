#!/bin/bash
# llama-7b confirmation phase, sequenced AFTER the qwen downstream grid frees its slices.
# 1) waits for the qwen c4_ds grid to finish (72 done-markers),
# 2) runs the full llama-7b PPL grid (wt2 then c4) across all MIG slices with PRISM_LOWMEM=1
#    (bool mask + uint8 codes + no dequant cache -> Sinkhorn fits a 24GB slice),
# 3) runs the FOCUSED llama-7b downstream grid (50%-sparsity decisive cells only).
# Detached launch:
#   cd /workspace/PRISM
#   setsid nohup bash scripts/run_llama_phase.sh >> logs/llama_phase.log 2>&1 </dev/null &
set -u
cd /workspace/PRISM
export PRISM_LOWMEM=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

QWEN_DS_DONE=results/ablation_qwen-0.5b_c4_ds/done
echo "[llama-phase] waiting for qwen downstream (72) $(date)"
for i in $(seq 1 600); do
    n=$(ls -1 "$QWEN_DS_DONE" 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -ge 72 ] && { echo "[llama-phase] qwen ds complete ($n) $(date)"; break; }
    sleep 60
done

# ---- Phase L1: full llama PPL grid (wt2 + c4), all techniques, all slices ----
for DS in wikitext2 c4; do
    echo "[llama-phase] ===== PPL dataset=$DS $(date) ====="
    ABL_MODEL=llama-7b ABL_DATASET="$DS" MAXROUNDS=60 PRISM_LOWMEM=1 \
        bash scripts/supervise_ablation.sh
done

# ---- Phase L2: FOCUSED llama downstream (50% sparsity cells only) ----
# Custom 3-sparsity->1 grid via ABL by reusing run_ablation_grid but limiting sparsities.
echo "[llama-phase] ===== focused downstream (50% cells) $(date) ====="
ABL_MODEL=llama-7b ABL_DATASET=c4 ABL_DS=1 PRISM_LOWMEM=1 \
ABL_TAG=ablation_llama-7b_c4_ds50 \
ABL_TECHS="abl-base abl-corr abl-extras abl-full sparsegpt sparsegpt-corr slim slim-corr jsq-wo jsq-wo-corr wanda-sinq" \
ABL_SPARS="0.50" \
DS_TASKS="hellaswag,arc_easy,arc_challenge,lambada,mmlu,mrr" DS_GEN_LIMIT=64 \
MAXROUNDS=60 \
    bash scripts/supervise_ablation.sh

echo "[llama-phase] ALL DONE $(date)"
