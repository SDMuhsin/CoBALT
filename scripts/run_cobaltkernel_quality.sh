#!/bin/bash
# cobaltkernel quality runner — reproduces the accel4bit lm_eval protocol EXACTLY (the
# scripts/run_accel4bit_ref.sh `bf16_quality` stage) for a "fake-quantized" bf16-format HF checkpoint of
# gemma-3-4b or medgemma-27b, so numbers are comparable row-for-row with results/accel4bit/BASELINE.md.
#
# Usage: scripts/run_cobaltkernel_quality.sh <model: gemma-3-4b|medgemma-27b> <hf_checkpoint_dir> <MIG-uuid> <tag> [tasks...]
#   tasks (optional, space-separated, name[:limit]): default protocol set
#     wikitext arc_easy medqa_4options pubmedqa medmcqa:1000
#   Writes results/cobaltkernel/<model>/<tag>/{eval_<task>.json, eval_summary.json, run.log, eval.log, run_meta.json}
#   Resume-safe: any task whose eval_<task>.json already exists in that dir is skipped (not re-run, not overwritten).
#   Launch detached: setsid nohup scripts/run_cobaltkernel_quality.sh <args> > /path/to/log 2>&1 & disown
#
# Offload: identical mechanism to accel4bit's medgemma-27b bf16 ref (REF27_OFFLOAD=prefetch in run_accel4bit_ref.sh /
# results/accel4bit/medgemma-27b/ref_bf16/NOTES.md) — vLLM's built-in CPU offloader, prefetch backend
# (offload_group_size=2, offload_num_in_group=1, offload_prefetch_step=1: every 2nd decoder layer lives in pinned
# host RAM and is asynchronously prefetched one layer ahead) + enforce_eager=True (torch.compile silently kills
# EngineCore under offload) + gpu_memory_utilization=0.90. Used ONLY when the checkpoint's safetensors bytes exceed
# 75% of the given MIG slice's total FB memory (54 GB bf16-sized 27B fakequant checkpoints on a 48 GB slice ->
# offload; a ~9 GB 4b checkpoint on a 24 GB 1g slice -> plain vLLM, exactly like the accel4bit gemma-3-4b bf16 ref).
set -u
MODEL=${1:?"usage: $0 <gemma-3-4b|medgemma-27b> <hf_checkpoint_dir> <MIG-uuid> <tag> [tasks...]"}
CKPT=${2:?hf_checkpoint_dir required}
MIG=${3:?MIG-uuid required}
TAG=${4:?tag required}
shift 4
TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(wikitext arc_easy medqa_4options pubmedqa medmcqa:1000)

case $MODEL in gemma-3-4b|medgemma-27b) ;; *) echo "unknown model $MODEL (want gemma-3-4b|medgemma-27b)"; exit 2 ;; esac
[ -d "$CKPT" ] || { echo "checkpoint dir not found: $CKPT"; exit 2; }

ROOT=/workspace/PTQResearch
ENV=/scratch/root/PTQResearch/env_accel_ref   # SAME venv as accel4bit ref (vllm 0.28.0, lm_eval 0.4.13) — required for row-for-row comparability
PY=$ENV/bin/python
OUT=$ROOT/results/cobaltkernel/$MODEL/$TAG
mkdir -p "$OUT"

unset PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME=/scratch/ckp908/prism_hf HF_HUB_CACHE=/scratch/ckp908/prism_hf/hub
export TMPDIR=/scratch/root/PTQResearch/tmp PIP_CONFIG_FILE=/dev/null
export CUDA_VISIBLE_DEVICES=$MIG
export CUDA_HOME=/scratch/root/PTQResearch/cuda-13
export PATH=/scratch/root/PTQResearch/cuda-13/bin:$ENV/bin:$PATH
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export TOKENIZERS_PARALLELISM=false
# NOTE: /root is at GPFS quota -> torch/vLLM/triton caches must live on /scratch
C=/scratch/root/PTQResearch/cache; mkdir -p $C/vllm $C/triton $C/inductor $C/xdg $C/nv
export VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg CUDA_CACHE_PATH=$C/nv
export VLLM_WORKER_MULTIPROC_METHOD=spawn     # lm_eval/driver touch CUDA before EngineCore starts -> fork fails otherwise

INCLUDE_PATH=$ROOT/scripts/accel4bit_lmeval_tasks   # shared pubmedqa override (parquet source)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run.log"; }

# ---- artifact size (bf16-format fakequant checkpoint on disk) ----
ARTIFACT_BYTES=$(find "$CKPT" -maxdepth 1 -name '*.safetensors' -printf '%s\n' 2>/dev/null | awk '{s+=$1} END{print s+0}')
ARTIFACT_GB=$(awk -v b="$ARTIFACT_BYTES" 'BEGIN{printf "%.2f", b/1e9}')

# ---- slice size (GB) from nvidia-smi -L, matched against the given MIG uuid (full uuid or short prefix both OK) ----
SLICE_LINE=$(nvidia-smi -L 2>/dev/null | grep -F "$MIG")
SLICE_GB=$(echo "$SLICE_LINE" | grep -oE '[0-9]+g\.[0-9]+gb' | grep -oE '\.[0-9]+gb' | tr -d '.gb')
if [ -z "${SLICE_GB:-}" ]; then
  log "WARNING: could not match MIG uuid '$MIG' in nvidia-smi -L output; assuming 24 GB slice (conservative)"
  SLICE_GB=24
fi

log "model=$MODEL ckpt=$CKPT mig=$MIG tag=$TAG artifact_gb=$ARTIFACT_GB slice_gb=$SLICE_GB tasks=${TASKS[*]}"

# ---- decide offload: same rule as accel4bit bf16-27b ref (bf16 54GB does not fit a 48GB/47.38GiB slice) ----
USE_OFFLOAD=$(awk -v a="$ARTIFACT_GB" -v s="$SLICE_GB" 'BEGIN{print (a > 0.75*s) ? 1 : 0}')
if [ "$USE_OFFLOAD" = "1" ]; then
  EXTRA="gpu_memory_utilization=0.90,enforce_eager=True,offload_group_size=${CKQ_GROUP:-2},offload_num_in_group=${CKQ_NUM:-1},offload_prefetch_step=${CKQ_STEP:-1}"
  log "OFFLOAD MODE: artifact ${ARTIFACT_GB} GB > 0.75x slice ${SLICE_GB} GB -> vLLM prefetch layer offload + enforce_eager (accel4bit bf16-27b mechanism, results/accel4bit/medgemma-27b/ref_bf16/NOTES.md)"
else
  EXTRA=""
  log "PLAIN MODE: artifact ${ARTIFACT_GB} GB fits slice ${SLICE_GB} GB -> plain vLLM (accel4bit gemma-3-4b bf16 mechanism)"
fi

VLLM_COMMON="max_model_len=4096,gpu_memory_utilization=0.85,dtype=bfloat16,enable_prefix_caching=False,seed=1234"
MA="$VLLM_COMMON${EXTRA:+,$EXTRA}"   # accel4bit_lmeval.py parses comma-list into a dict left-to-right -> EXTRA's
                                     # gpu_memory_utilization=0.90 overrides VLLM_COMMON's 0.85 when offloading

# ---- resume-safe task filter ----
RUN_TASKS=()
for spec in "${TASKS[@]}"; do
  name=${spec%%:*}
  if [ -f "$OUT/eval_${name}.json" ]; then
    log "SKIP $name (eval_${name}.json already exists)"
  else
    RUN_TASKS+=("$spec")
  fi
done

WALL=0
EXIT=0
if [ ${#RUN_TASKS[@]} -eq 0 ]; then
  log "all requested tasks already have results in $OUT; nothing to run"
else
  log "lm_eval (vllm) $CKPT -> $OUT  model_args=$MA tasks=${RUN_TASKS[*]}"
  T0=$(date +%s)
  $PY $ROOT/src/accel4bit_lmeval.py --backend vllm --pretrained "$CKPT" --out_dir "$OUT" --model_args "$MA" \
      --batch_size auto --seed 1234 --include_path "$INCLUDE_PATH" --tasks "${RUN_TASKS[@]}" >> "$OUT/eval.log" 2>&1
  EXIT=$?
  T1=$(date +%s)
  WALL=$((T1-T0))
  log "quality_exit=$EXIT wall_s=$WALL"
fi

OFFLOAD_JSON=false; [ "$USE_OFFLOAD" = "1" ] && OFFLOAD_JSON=true
cat > "$OUT/run_meta.json" <<EOF
{"model": "$MODEL", "tag": "$TAG", "ckpt": "$CKPT", "mig": "$MIG", "slice_gb": $SLICE_GB,
 "artifact_gb": $ARTIFACT_GB, "artifact_bytes": $ARTIFACT_BYTES, "offload": $OFFLOAD_JSON,
 "model_args": "$MA", "tasks_requested": "${TASKS[*]}", "tasks_run_this_call": "${RUN_TASKS[*]:-}",
 "wall_s_this_call": $WALL, "exit_this_call": $EXIT, "timestamp": "$(date -u +%FT%TZ)"}
EOF
log "wrote $OUT/run_meta.json"

# ---- one-line-per-task comparison vs bf16 and gguf rows in results/accel4bit/BASELINE.json ----
log "comparison vs results/accel4bit/BASELINE.json ($MODEL bf16 / gguf rows):"
$PY $ROOT/src/cobaltkernel/collect_quality.py --model "$MODEL" --tag "$TAG" --compare-only 2>&1 | tee -a "$OUT/run.log"

log "DONE $MODEL $TAG (results: $OUT)"
