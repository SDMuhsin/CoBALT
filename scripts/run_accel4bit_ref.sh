#!/bin/bash
# accel4bit REFERENCE arm: bf16 quality (+ bf16 speed where it fits) and FP8-dynamic speed+quality references.
# Usage: scripts/run_accel4bit_ref.sh <gemma-3-4b|medgemma-27b> [MIG-UUID] [stage ...]
#   stages gemma-3-4b : extract_text bf16_quality bf16_speed                 (default: all three)
#   stages medgemma-27b: fp8_download fp8_quality fp8_speed bf16_quality      (default: all four, in this order)
# Re-runnable: each stage writes to results/accel4bit/<model>/ref_<x>/ and skips nothing (overwrites) — delete
# nothing by hand; every run is timestamped in the log. See results/accel4bit/<model>/ref_*/NOTES.md.
set -u
MODEL=${1:?model-short}
MIG=${2:-MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0}    # PROTOCOL: REF slice = the 2g.48gb slice
shift 2 2>/dev/null || shift $#
STAGES=("$@")

ROOT=/workspace/PTQResearch
ENV=/scratch/root/PTQResearch/env_accel_ref                # own venv: vllm==0.28.0 + lm_eval (never the shared one)
PY=$ENV/bin/python
MODELS=/scratch/root/PTQResearch/accel4bit_models/$MODEL
RES=$ROOT/results/accel4bit/$MODEL
mkdir -p "$RES" "$MODELS"

unset PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME=/scratch/ckp908/prism_hf HF_HUB_CACHE=/scratch/ckp908/prism_hf/hub
export TMPDIR=/scratch/root/PTQResearch/tmp PIP_CONFIG_FILE=/dev/null
export CUDA_VISIBLE_DEVICES=$MIG
export CUDA_HOME=/scratch/root/PTQResearch/cuda-13
export PATH=/scratch/root/PTQResearch/cuda-13/bin:$ENV/bin:$PATH
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO}
export TOKENIZERS_PARALLELISM=false
# NOTE: /root is at GPFS quota -> all torch/vLLM/triton caches must live on /scratch (else Errno 122 kills EngineCore)
C=/scratch/root/PTQResearch/cache; mkdir -p $C/vllm $C/triton $C/inductor $C/xdg $C/nv
export VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg CUDA_CACHE_PATH=$C/nv
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # lm_eval/driver touch CUDA before EngineCore starts -> fork fails ("Cannot re-initialize CUDA in forked subprocess")

TASKS_PRIORITY="wikitext medqa_4options arc_easy pubmedqa medmcqa:1000"   # PROTOCOL tasks; priority order for slow bf16-27B
INCLUDE_PATH=$ROOT/scripts/accel4bit_lmeval_tasks   # shared pubmedqa override (parquet source; stock task broken under datasets 5.x)
VLLM_COMMON="max_model_len=4096,gpu_memory_utilization=0.85,dtype=bfloat16,enable_prefix_caching=False,seed=1234"

log() { echo "[$(date '+%F %T')] $*"; }

speed_bench() {  # $1=model dir, $2=out dir, $3=extra vllm args (comma-free CLI flags)
  local M=$1 OUT=$2; shift 2
  mkdir -p "$OUT"
  log "vllm bench latency -> $OUT/speed_latency.json"
  VLLM_LOGGING_LEVEL=DEBUG vllm bench latency --model "$M" --dtype bfloat16 --max-model-len 4096 \
      --input-len 512 --output-len 128 --batch-size 1 --num-iters 10 --num-iters-warmup 3 \
      --gpu-memory-utilization 0.85 --seed 1234 --output-json "$OUT/speed_latency.json" "$@" > "$OUT/speed_latency.log" 2>&1
  echo "latency_exit=$?" | tee -a "$OUT/speed_latency.log"
  log "vllm bench throughput -> $OUT/speed_throughput.json"
  vllm bench throughput --model "$M" --dtype bfloat16 --max-model-len 4096 \
      --input-len 512 --output-len 128 --num-prompts 256 --max-num-seqs 32 \
      --gpu-memory-utilization 0.85 --seed 1234 --output-json "$OUT/speed_throughput.json" "$@" > "$OUT/speed_throughput.log" 2>&1
  echo "throughput_exit=$?" | tee -a "$OUT/speed_throughput.log"
  # TTFT / per-token decode breakdown (single stream, greedy), same 512->128 shape
  $PY $ROOT/src/accel4bit_ttft.py --model "$M" --out "$OUT/speed_ttft.json" "$@" > "$OUT/speed_ttft.log" 2>&1
  echo "ttft_exit=$?" | tee -a "$OUT/speed_ttft.log"
}

quality_vllm() {  # $1=model dir, $2=out dir, $3=extra model_args (comma-joined), $4=tasks
  local M=$1 OUT=$2 EXTRA=${3:-} TASKS=${4:-$TASKS_PRIORITY}
  mkdir -p "$OUT"
  local MA="$VLLM_COMMON${EXTRA:+,$EXTRA}"
  log "lm_eval (vllm) $M -> $OUT  model_args=$MA tasks=$TASKS"
  $PY $ROOT/src/accel4bit_lmeval.py --backend vllm --pretrained "$M" --out_dir "$OUT" --model_args "$MA" \
      --batch_size auto --seed 1234 --include_path "$INCLUDE_PATH" --tasks $TASKS >> "$OUT/eval.log" 2>&1
  echo "quality_exit=$?" | tee -a "$OUT/eval.log"
}

case $MODEL in
gemma-3-4b)
  [ ${#STAGES[@]} -eq 0 ] && STAGES=(extract_text bf16_quality bf16_speed)
  TEXT=$MODELS/text_bf16
  for st in "${STAGES[@]}"; do case $st in
    extract_text)
      # PROTOCOL: 4b ckpt is multimodal; evaluate the TEXT model only -> extract language_model into Gemma3ForCausalLM
      CUDA_VISIBLE_DEVICES="" $PY $ROOT/src/accel4bit_extract_text.py unsloth/gemma-3-4b-it "$TEXT" 2>&1 | tee "$RES/extract_text.log" ;;
    bf16_quality) quality_vllm "$TEXT" "$RES/ref_bf16" "" ;;
    bf16_speed)   speed_bench "$TEXT" "$RES/ref_bf16" ;;
    *) echo "unknown stage $st"; exit 2 ;;
  esac; done ;;
medgemma-27b)
  [ ${#STAGES[@]} -eq 0 ] && STAGES=(fp8_download fp8_quality fp8_speed bf16_quality)
  BF16=/scratch/ckp908/prism_hf/hub/models--unsloth--medgemma-27b-text-it/snapshots/b780610baf99c087ba3719a77cf0dacec7261a65
  FP8_REPO=turnio/medgemma-27b-text-it-FP8-Dynamic     # existing ungated FP8_DYNAMIC (llmcompressor 0.13.0, ignore=[lm_head])
  FP8=$MODELS/ref_fp8
  for st in "${STAGES[@]}"; do case $st in
    fp8_download)
      $PY - <<EOF 2>&1 | tee "$RES/ref_fp8_download.log"
import os, json
from huggingface_hub import snapshot_download
p = snapshot_download("$FP8_REPO", max_workers=8)
os.makedirs("$MODELS", exist_ok=True)
if os.path.islink("$FP8") or os.path.exists("$FP8"):
    os.remove("$FP8") if os.path.islink("$FP8") else None
if not os.path.exists("$FP8"):
    os.symlink(p, "$FP8")
print("FP8 snapshot:", p, "->", "$FP8")
print("quantization_config:", json.dumps(json.load(open(p + "/config.json"))["quantization_config"]))
EOF
      ;;
    fp8_quality) quality_vllm "$FP8" "$RES/ref_fp8" "" ;;
    fp8_speed)   speed_bench "$FP8" "$RES/ref_fp8" ;;
    bf16_quality)
      # bf16 27B (54 GB) does not fit the 48 GB slice: vLLM prefetch layer offload (async H2D) first,
      # fallback UVA cpu_offload_gb, fallback HF device_map=auto. Label in NOTES.md which one produced the numbers.
      OUT=$RES/ref_bf16; mkdir -p "$OUT"
      MODE=${REF27_OFFLOAD:-prefetch}
      case $MODE in
        prefetch) EXTRA="gpu_memory_utilization=0.90,enforce_eager=True,offload_group_size=${REF27_GROUP:-2},offload_num_in_group=${REF27_NUM:-1},offload_prefetch_step=${REF27_STEP:-1}";;   # enforce_eager: with torch.compile the EngineCore dies silently during inductor compile (measured 21:49-21:51)
        uva)      EXTRA="gpu_memory_utilization=0.90,enforce_eager=True,cpu_offload_gb=${REF27_CPU_GB:-24}";;
        hf)       EXTRA="";;
      esac
      if [ "$MODE" = hf ]; then
        log "lm_eval (hf, device_map=auto offload) $BF16 -> $OUT"
        $PY $ROOT/src/accel4bit_lmeval.py --backend hf --pretrained "$BF16" --out_dir "$OUT" \
            --model_args "dtype=bfloat16,parallelize=True,max_memory_per_gpu=40GiB,max_cpu_memory=200GiB,offload_folder=$TMPDIR/hf_offload" \
            --batch_size 4 --seed 1234 --include_path "$INCLUDE_PATH" --tasks $TASKS_PRIORITY >> "$OUT/eval.log" 2>&1
        echo "quality_exit=$?" | tee -a "$OUT/eval.log"
      else
        # gpu_memory_utilization is overridden by EXTRA (later key wins in the parser)
        quality_vllm "$BF16" "$OUT" "$EXTRA"
      fi ;;
    *) echo "unknown stage $st"; exit 2 ;;
  esac; done ;;
*) echo "unknown model $MODEL"; exit 2 ;;
esac
log "DONE $MODEL stages: ${STAGES[*]}"
