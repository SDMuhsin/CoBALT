#!/usr/bin/env bash
# accel4bit ARM A (gptq): GPTQ W4A16 sym g128 (llm-compressor 0.13.0) -> vLLM 0.28.0 gptq_marlin.
# Usage: scripts/run_accel4bit_gptq.sh <gemma-3-4b|medgemma-27b> [MIG-UUID]
#   STAGES="quant sanity quality speed bpw" (env, default all) selects stages.
# Long runs: setsid nohup bash scripts/run_accel4bit_gptq.sh medgemma-27b > results/accel4bit/medgemma-27b/gptq/run.log 2>&1 < /dev/null & disown
set -uo pipefail
MODEL=${1:?model short name}
MIG=${2:-MIG-bef7a31e-4317-582c-a97d-75e9e429441c}
STAGES=${STAGES:-"quant sanity quality speed bpw"}
REPO=/workspace/PTQResearch
ART=/scratch/root/PTQResearch/accel4bit_models/$MODEL/gptq
RES=$REPO/results/accel4bit/$MODEL/gptq
mkdir -p "$ART" "$RES"

# --- environment ---
unset PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME=/scratch/ckp908/prism_hf HF_HUB_CACHE=/scratch/ckp908/prism_hf/hub
export TMPDIR=/scratch/root/PTQResearch/tmp PIP_CONFIG_FILE=/dev/null
export CUDA_VISIBLE_DEVICES=$MIG
export CUDA_HOME=/scratch/root/PTQResearch/cuda-13
export PATH=/scratch/root/PTQResearch/cuda-13/bin:/scratch/root/PTQResearch/env_accel_gptq_arm/bin:/scratch/root/PTQResearch/env_accel_vllm/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0   # avoid FlashInfer sampler JIT; greedy decoding anyway
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # sanity attempt 1: "Cannot re-initialize CUDA in forked subprocess" (EngineCore forked)
export TOKENIZERS_PARALLELISM=false
# Compile/JIT caches MUST live on /scratch: /root (GPFS home) sits at its user quota and vLLM's inductor autotune died with
# "[Errno 122] Disk quota exceeded" (gemma-3-4b sanity attempts 2-3, see results/accel4bit/gemma-3-4b/gptq/NOTES.md). Applies to all arms.
C=/scratch/root/PTQResearch/cache
mkdir -p $C/vllm $C/triton $C/inductor $C/xdg $C/nv
export VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg CUDA_CACHE_PATH=$C/nv
PY=/scratch/root/PTQResearch/env_accel_gptq_arm/bin/python      # = env_accel_vllm site-packages + lm_eval 0.4.13
VLLM=/scratch/root/PTQResearch/env_accel_vllm/bin/vllm
LMEVAL="/scratch/root/PTQResearch/env_accel_gptq_arm/bin/lm_eval --include_path $REPO/scripts/accel4bit_lmeval_tasks"   # shared pubmedqa override (parquet branch; datasets 5 rejects bigbio script)
GPU_UTIL=${GPU_UTIL:-0.85}   # 4b: 0.85; 27B on a 1g.24gb slice needs 0.95 (15.47 GiB weights leave 0.25 GiB KV at 0.85)
VARGS="pretrained=$ART,quantization=compressed-tensors,max_model_len=4096,gpu_memory_utilization=$GPU_UTIL,dtype=bfloat16,enable_prefix_caching=False,seed=1234"
EAGER=${EAGER:-0}   # EAGER=1: enforce_eager (no CUDA graphs). 27B on 1g.24gb at 0.95: KV 3.24 GiB + non-KV 19.83 GiB leave nothing for graph capture (CUDA OOM 672 MiB)
if [ "$EAGER" = 1 ]; then VARGS="$VARGS,enforce_eager=True"; EAGER_ARGS="--enforce-eager"; else EAGER_ARGS=""; fi

log() { echo "[$(date '+%F %T')] $*"; }
gpu_idle_check() {
  local used; used=$(nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$MIG" || true)
  log "slice $MIG compute-apps on slice: $used (expect 0)"; nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader | grep "$MIG" || true
}
# background peak-memory sampler: nvidia-smi compute-apps reports the PARENT GPU uuid (not the MIG uuid), so we sum
# used_memory over the processes descending from this runner shell (= the vLLM engine procs on our slice). Max MiB -> $1
descendants() { echo "$1"; local c; for c in $(pgrep -P "$1" 2>/dev/null); do descendants "$c"; done; }
mem_sampler() { local out=$1; ( : > "$out.samples"; while true; do local pids; pids=$(descendants $$ | tr '\n' '|' | sed 's/|$//');
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v re="^(${pids})$" '$1 ~ re {s+=$2} END{print s+0}' >> "$out.samples"; sleep 2; done ) >/dev/null 2>&1 < /dev/null & echo $!; }
stop_sampler() { kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; sort -n "$2.samples" | tail -1 > "$2"; log "peak slice memory (MiB): $(cat "$2")"; }

log "MODEL=$MODEL MIG=$MIG STAGES=$STAGES ART=$ART RES=$RES"
gpu_idle_check

for STAGE in $STAGES; do
case $STAGE in
quant)
  log "=== STAGE quant ==="
  $PY "$REPO/src/accel4bit_gptq_quant.py" "$MODEL" --out "$ART" 2>&1 | tee "$RES/quant.log"
  grep -q QUANT_OK "$RES/quant.log" || { log "QUANT FAILED"; exit 1; }
  ls -la "$ART" | tee -a "$RES/quant.log"
  ;;
sanity)
  log "=== STAGE sanity (vLLM generation + kernel dispatch from DEBUG log) ==="
  S=$(mem_sampler "$RES/peak_mem_sanity.txt")
  VLLM_LOGGING_LEVEL=DEBUG $PY "$REPO/src/accel4bit_vllm_single_stream.py" --model "$ART" --sanity-only --gpu-memory-utilization $GPU_UTIL $EAGER_ARGS \
      --out-json "$RES/sanity.json" > "$RES/sanity.log" 2>&1
  stop_sampler "$S" "$RES/peak_mem_sanity.txt"
  grep -q SANITY_OK "$RES/sanity.log" || { log "SANITY FAILED"; tail -40 "$RES/sanity.log"; exit 1; }
  grep -iE "marlin|machete|cutlass|scaled_mm|kernel" "$RES/sanity.log" | sort | uniq -c | sort -rn | head -20 | tee "$RES/kernel_dispatch.txt"
  grep "\[sanity\]" "$RES/sanity.log"
  ;;
quality)
  log "=== STAGE quality (lm_eval vllm backend, seed 1234) ==="
  $LMEVAL --model vllm --model_args "$VARGS" --tasks wikitext,arc_easy,medqa_4options,pubmedqa --seed 1234 \
      --batch_size auto --output_path "$RES/lm_eval_main" > "$RES/eval_main.log" 2>&1
  log "main tasks exit=$?"
  $LMEVAL --model vllm --model_args "$VARGS" --tasks medmcqa --limit 1000 --seed 1234 \
      --batch_size auto --output_path "$RES/lm_eval_medmcqa" > "$RES/eval_medmcqa.log" 2>&1
  log "medmcqa exit=$?"
  $PY "$REPO/src/accel4bit_split_lmeval.py" "$RES"
  ;;
speed)
  log "=== STAGE speed (vLLM native benches + single-stream TTFT script) ==="
  S=$(mem_sampler "$RES/peak_mem_speed.txt")
  VLLM_LOGGING_LEVEL=DEBUG $VLLM bench latency --model "$ART" --quantization compressed-tensors --dtype bfloat16 \
      --max-model-len 4096 --gpu-memory-utilization $GPU_UTIL $EAGER_ARGS --no-enable-prefix-caching \
      --input-len 512 --output-len 128 --batch-size 1 --num-iters 10 --num-iters-warmup 3 \
      --output-json "$RES/speed_latency.json" > "$RES/speed_latency.log" 2>&1
  log "bench latency exit=$?"
  $PY "$REPO/src/accel4bit_vllm_single_stream.py" --model "$ART" --input-len 512 --output-len 128 --iters 10 --warmup 3 --gpu-memory-utilization $GPU_UTIL $EAGER_ARGS \
      --out-json "$RES/speed_single_stream.json" > "$RES/speed_single_stream.log" 2>&1
  log "single-stream exit=$?"
  $VLLM bench throughput --model "$ART" --quantization compressed-tensors --dtype bfloat16 \
      --max-model-len 4096 --gpu-memory-utilization $GPU_UTIL $EAGER_ARGS --no-enable-prefix-caching --max-num-seqs 32 \
      --input-len 512 --output-len 128 --num-prompts 256 --seed 1234 \
      --output-json "$RES/speed_throughput.json" > "$RES/speed_throughput.log" 2>&1
  log "bench throughput exit=$?"
  stop_sampler "$S" "$RES/peak_mem_speed.txt"
  grep -ihE "marlin|machete|cutlass|scaled_mm" "$RES/speed_latency.log" | sort | uniq -c | sort -rn | head -10 | tee "$RES/kernel_dispatch_speed.txt"
  grep -hE "Avg latency|Throughput|tok/s|TTFT|single-stream" "$RES"/speed_*.log
  ;;
bpw)
  log "=== STAGE bpw ==="
  $PY "$REPO/src/accel4bit_bpw.py" "$ART" --out-json "$RES/bpw.json" | tee "$RES/bpw.log"
  ;;
*) log "unknown stage $STAGE"; exit 2;;
esac
done
log "ALL_STAGES_DONE"
