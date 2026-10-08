#!/usr/bin/env bash
# accel4bit PHASE 3: cross-arm SPEED benchmark on ONE slice (PROTOCOL: the 2g.48gb slice MIG-1d47bdbe..., because the
# FP8 27B reference only fits there). Every arm is run SEQUENTIALLY, one process at a time, in that arm's own runtime/venv.
#
# Usage: scripts/run_accel4bit_speed_oneslice.sh <gemma-3-4b|medgemma-27b> <MIG-UUID> [arm ...]
#   arms (default = every arm whose artifact exists): bf16 (4b only) ref_fp8 (27b only) gptq awq nvfp4 gguf
#   gguf on medgemma-27b also benchmarks the unsloth prebuilt GGUFs (tags gguf_unsloth_q4km, gguf_unsloth_udq4kxl) if present.
#   env: GPU_UTIL=0.85  SKIP_CUBLAS=0  OUT_SUBDIR=speed_oneslice
# Long run: setsid nohup bash scripts/run_accel4bit_speed_oneslice.sh gemma-3-4b MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0 \
#           > results/accel4bit/gemma-3-4b/speed_oneslice/run.log 2>&1 < /dev/null & disown
# Outputs: results/accel4bit/<model>/speed_oneslice/<arm>.json (+ raw logs/JSONs per arm) and SPEED_TABLE.md
set -uo pipefail
MODEL=${1:?model short name}
MIG=${2:?MIG-UUID (PROTOCOL phase 3: MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0)}
shift 2
ARMS=("$@")
ROOT=/workspace/PTQResearch
MODELS=/scratch/root/PTQResearch/accel4bit_models/$MODEL
RES=$ROOT/results/accel4bit/$MODEL/${OUT_SUBDIR:-speed_oneslice}
GPU_UTIL=${GPU_UTIL:-0.85}
mkdir -p "$RES"
SUMM="$ROOT/src/accel4bit_speed_oneslice.py"

# ---------------- environment (union of the arms' runners) ----------------
unset PYTHONPATH LIBRARY_PATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME=/scratch/ckp908/prism_hf HF_HUB_CACHE=/scratch/ckp908/prism_hf/hub
export TMPDIR=/scratch/root/PTQResearch/tmp/p3          # short path: vLLM ZMQ ipc sockets live in TMPDIR (sun_path <= 107 chars)
CACHE=/scratch/root/PTQResearch/cache; mkdir -p "$TMPDIR" $CACHE/{vllm,triton,inductor,xdg,nv}
export VLLM_CACHE_ROOT=$CACHE/vllm TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
       XDG_CACHE_HOME=$CACHE/xdg CUDA_CACHE_PATH=$CACHE/nv FLASHINFER_WORKSPACE_BASE=$CACHE/xdg
export CUDA_VISIBLE_DEVICES=$MIG
export CUDA_HOME=/scratch/root/PTQResearch/cuda-13
export PATH=/scratch/root/PTQResearch/cuda-13/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false PIP_CONFIG_FILE=/dev/null
export VLLM_LOGGING_LEVEL=INFO   # kernel-selection lines ("Using MarlinLinearKernel for ...", "Selected CutlassFP8...", "... for NVFP4 GEMM") are INFO
LC=$ROOT/temp/llama.cpp; BIN=$LC/build/bin; BIN_CUBLAS=$LC/build-cublas/bin
CUDA13=/scratch/root/PTQResearch/cuda-13
NTHREADS=${NTHREADS:-16}
PY_REF=/scratch/root/PTQResearch/env_accel_ref/bin/python
PY_SUMM=$PY_REF

log() { echo "[$(date '+%F %T')] $*"; }
SLICE_TYPE=$(nvidia-smi -L 2>/dev/null | grep -F "$MIG" | sed -E 's/.*MIG ([0-9]+g\.[0-9]+gb).*/\1/' | head -1); SLICE_TYPE=${SLICE_TYPE:-unknown}

# --- peak GPU memory sampler: sum nvidia-smi compute-apps used_memory over processes descending from this shell (1 s) ---
descendants() { echo "$1"; local c; for c in $(pgrep -P "$1" 2>/dev/null); do descendants "$c"; done; }
sampler_start() { local out=$1; : > "$out"; ( while true; do pids=$(descendants $$ | tr '\n' '|' | sed 's/|$//');
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v re="^(${pids})$" '$1 ~ re {s+=$2} END{print s+0}' >> "$out"; sleep 1; done ) >/dev/null 2>&1 < /dev/null & echo $!; }
sampler_stop() { kill "$1" 2>/dev/null; wait "$1" 2>/dev/null; }

slice_idle_check() {
  # nvidia-smi compute-apps reports the parent GPU uuid, so check per-process CUDA_VISIBLE_DEVICES instead.
  # Rule: other arms may legitimately use the 48 GB slice (e.g. NVFP4 27B quality eval) -> if busy, POLL every
  # 2 min with a liveness check on the occupying pid; NEVER kill a process that is not ours. WAIT_IDLE=0 disables waiting.
  while true; do
    local busy=0 p
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=$MIG"; then
        busy=1; log "slice $MIG busy: pid $p (alive=$(kill -0 $p 2>/dev/null && echo yes || echo no)) $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)"
      fi
    done
    if [ $busy = 0 ]; then log "slice $MIG ($SLICE_TYPE) is idle (no compute process bound to it)"; return 0; fi
    [ "${WAIT_IDLE:-1}" = 1 ] || { log "WAIT_IDLE=0 -> proceeding although the slice is busy (numbers would be confounded)"; return 0; }
    sleep 120
  done
}

# ---------------- vLLM arm: latency (e2e) -> TTFT/decode split -> throughput ----------------
# run_vllm <arm> <python> <artifact> [extra vllm CLI args...]
run_vllm() {
  local ARM=$1 PY=$2 ART=$3; shift 3
  local EXTRA=("$@")
  log "=== ARM $ARM (vLLM) artifact=$ART python=$PY extra=${EXTRA[*]:-}"
  export PATH=$(dirname "$PY"):/scratch/root/PTQResearch/cuda-13/bin:$PATH   # arm venv bin first: FlashInfer/torch JIT need ninja + nvcc (nvfp4 failed with "No such file: ninja" without it)
  local COMMON=(--model "$ART" --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization "$GPU_UTIL" --no-enable-prefix-caching --seed 1234 "${EXTRA[@]}")
  local S; S=$(sampler_start "$RES/${ARM}_peakmem.samples")
  log "[$ARM] vllm bench latency (512->128, bs1, 10 iters + 3 warmup)"
  "$PY" -m vllm.entrypoints.cli.main bench latency "${COMMON[@]}" --input-len 512 --output-len 128 --batch-size 1 \
      --num-iters 10 --num-iters-warmup 3 --output-json "$RES/${ARM}_latency.json" > "$RES/${ARM}_latency.log" 2>&1
  log "[$ARM] latency exit=$? $(grep -h 'Avg latency' "$RES/${ARM}_latency.log" | tail -1)"
  log "[$ARM] TTFT / decode split (src/accel4bit_ttft.py)"
  "$PY" "$ROOT/src/accel4bit_ttft.py" --model "$ART" --out "$RES/${ARM}_ttft.json" --input-len 512 --output-len 128 --iters 10 --warmup 3 \
      --gpu-memory-utilization "$GPU_UTIL" "${EXTRA[@]}" > "$RES/${ARM}_ttft.log" 2>&1
  log "[$ARM] ttft exit=$? $(grep -hE '"(ttft_s_median|decode_tok_s)"' "$RES/${ARM}_ttft.log" | tr -d '\n ')"
  log "[$ARM] vllm bench throughput (256 x 512->128, max-num-seqs 32)"
  "$PY" -m vllm.entrypoints.cli.main bench throughput "${COMMON[@]}" --max-num-seqs 32 --input-len 512 --output-len 128 --num-prompts 256 \
      --output-json "$RES/${ARM}_throughput.json" > "$RES/${ARM}_throughput.log" 2>&1
  log "[$ARM] throughput exit=$? $(grep -h 'Throughput' "$RES/${ARM}_throughput.log" | tail -1)"
  sampler_stop "$S"
  local CMDS="python -m vllm.entrypoints.cli.main bench latency ${COMMON[*]} --input-len 512 --output-len 128 --batch-size 1 --num-iters 10 --num-iters-warmup 3 ; python src/accel4bit_ttft.py --model $ART ${EXTRA[*]:-} ; python -m vllm.entrypoints.cli.main bench throughput ${COMMON[*]} --max-num-seqs 32 --input-len 512 --output-len 128 --num-prompts 256"
  "$PY_SUMM" "$SUMM" vllm --arm "$ARM" --model "$MODEL" --slice "$MIG" --slice-type "$SLICE_TYPE" --outdir "$RES" --artifact "$ART" \
      --latency-json "$RES/${ARM}_latency.json" --ttft-json "$RES/${ARM}_ttft.json" --throughput-json "$RES/${ARM}_throughput.json" \
      --latency-log "$RES/${ARM}_latency.log" --ttft-log "$RES/${ARM}_ttft.log" --throughput-log "$RES/${ARM}_throughput.log" \
      --peak-samples "$RES/${ARM}_peakmem.samples" --python "$PY" --extra-note "commands: $CMDS"
}

# ---------------- llama.cpp arm: llama-bench (MMQ + forced-cuBLAS build) + llama-batched-bench ----------------
# run_gguf <tag-arm-name> <gguf file>
run_gguf() {
  local ARM=$1 G=$2
  log "=== ARM $ARM (llama.cpp) gguf=$G"
  export LD_LIBRARY_PATH=$CUDA13/lib:/.singularity.d/libs
  local S; S=$(sampler_start "$RES/${ARM}_peakmem.samples")
  log "[$ARM] llama-bench -p 512 -n 128 -ngl 99 -r 5 (MMQ build)"
  "$BIN/llama-bench" -m "$G" -ngl 99 -p 512 -n 128 -r 5 -t $NTHREADS -o json > "$RES/${ARM}_bench_mmq.json" 2> "$RES/${ARM}_bench_mmq.stderr"
  log "[$ARM] bench exit=$?"
  if [ "${SKIP_CUBLAS:-0}" = 0 ] && [ -x "$BIN_CUBLAS/llama-bench" ]; then
    log "[$ARM] llama-bench forced-cuBLAS counterfactual build"
    "$BIN_CUBLAS/llama-bench" -m "$G" -ngl 99 -p 512 -n 128 -r 5 -t $NTHREADS -o json > "$RES/${ARM}_bench_cublas.json" 2> "$RES/${ARM}_bench_cublas.stderr"
    log "[$ARM] cublas bench exit=$?"
  fi
  log "[$ARM] llama-batched-bench -npp 512 -ntg 128 -npl 1,8,32"
  "$BIN/llama-batched-bench" -m "$G" -ngl 99 -npp 512 -ntg 128 -npl 1,8,32 -c 32768 -b 4096 -ub 512 -t $NTHREADS > "$RES/${ARM}_batched.log" 2>&1
  log "[$ARM] batched exit=$?"; grep -E '^\|' "$RES/${ARM}_batched.log"
  sampler_stop "$S"
  unset LD_LIBRARY_PATH
  "$PY_SUMM" "$SUMM" gguf --arm "$ARM" --model "$MODEL" --slice "$MIG" --slice-type "$SLICE_TYPE" --outdir "$RES" --artifact "$G" \
      --bench-json "$RES/${ARM}_bench_mmq.json" --bench-cublas-json "$RES/${ARM}_bench_cublas.json" --bench-stderr "$RES/${ARM}_bench_mmq.stderr" \
      --batched-log "$RES/${ARM}_batched.log" --peak-samples "$RES/${ARM}_peakmem.samples" \
      --extra-note "commands: llama-bench -m $G -ngl 99 -p 512 -n 128 -r 5 -t $NTHREADS ; llama-batched-bench -m $G -ngl 99 -npp 512 -ntg 128 -npl 1,8,32 -c 32768 -b 4096 -ub 512 -t $NTHREADS"
}

pending() { "$PY_SUMM" "$SUMM" pending --arm "$1" --model "$MODEL" --slice "$MIG" --slice-type "$SLICE_TYPE" --outdir "$RES" --artifact "${3:-}" --reason "$2"; }

# ---------------- arm table: artifact, venv, extra CLI flags ----------------
case $MODEL in
  gemma-3-4b)   GGUF_NAME=gemma-3-4b-it; DEFAULT_ARMS=(bf16 gptq awq nvfp4 gguf) ;;
  medgemma-27b) GGUF_NAME=medgemma-27b-text-it; DEFAULT_ARMS=(ref_fp8 gptq awq nvfp4 gguf) ;;
  biomistral-7b) GGUF_NAME=biomistral-7b; DEFAULT_ARMS=(gguf) ;;   # CoBALT arms: scripts/run_biomistral_speed.sh
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
[ ${#ARMS[@]} -eq 0 ] && ARMS=("${DEFAULT_ARMS[@]}")

log "MODEL=$MODEL MIG=$MIG ($SLICE_TYPE) ARMS=${ARMS[*]} RES=$RES GPU_UTIL=$GPU_UTIL"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader | head -2
slice_idle_check

for ARM in "${ARMS[@]}"; do
  case $ARM in
    bf16)
      ART=$MODELS/text_bf16
      if [ "$MODEL" != gemma-3-4b ]; then pending bf16 "bf16 $MODEL (54 GB) does not fit any slice; no bf16 speed possible (see ref_bf16/NOTES.md)" "$ART"
      elif [ -f "$ART/config.json" ]; then run_vllm bf16 "$PY_REF" "$ART"
      else pending bf16 "artifact $ART missing" "$ART"; fi ;;
    ref_fp8)
      ART=$MODELS/ref_fp8
      if [ -f "$ART/config.json" ]; then run_vllm ref_fp8 "$PY_REF" "$ART"
      else pending ref_fp8 "artifact $ART missing" "$ART"; fi ;;
    gptq)
      ART=$MODELS/gptq
      if [ -f "$ART/config.json" ] && ls "$ART"/*.safetensors >/dev/null 2>&1; then
        run_vllm gptq /scratch/root/PTQResearch/env_accel_gptq_arm/bin/python "$ART" --quantization compressed-tensors
      else pending gptq "artifact $ART missing/incomplete (quantization not finished)" "$ART"; fi ;;
    awq)
      ART=$MODELS/awq
      if [ -f "$ART/config.json" ] && ls "$ART"/*.safetensors >/dev/null 2>&1; then
        # the AWQ artifact keeps the multimodal (Gemma3ForConditionalGeneration) config -> disable image profiling like the arm's own runner
        if grep -q ConditionalGeneration "$ART/config.json"; then MM=(--limit-mm-per-prompt '{"image":0}'); else MM=(); fi
        run_vllm awq /scratch/root/PTQResearch/env_accel_awq/bin/python "$ART" --quantization compressed-tensors "${MM[@]}"
      else pending awq "artifact $ART missing/incomplete (quantization not finished)" "$ART"; fi ;;
    nvfp4)
      ART=$MODELS/nvfp4
      if [ -f "$ART/config.json" ] && ls "$ART"/*.safetensors >/dev/null 2>&1; then
        run_vllm nvfp4 /scratch/root/PTQResearch/env_accel_nvfp4/bin/python "$ART" --quantization compressed-tensors
      else pending nvfp4 "artifact $ART missing/incomplete (quantization not finished)" "$ART"; fi ;;
    gguf)
      G=$MODELS/gguf/$GGUF_NAME-Q4_K_M.gguf
      if [ -f "$G" ]; then run_gguf gguf "$G"; else pending gguf "own Q4_K_M $G missing (imatrix/quantize not finished)" "$G"; fi
      if [ "$MODEL" = medgemma-27b ]; then
        for t in q4km:medgemma-27b-text-it-Q4_K_M.gguf udq4kxl:medgemma-27b-text-it-UD-Q4_K_XL.gguf; do
          P=$MODELS/gguf/unsloth_prebuilt/${t#*:}
          if [ -f "$P" ]; then run_gguf gguf_unsloth_${t%%:*} "$P"; else pending gguf_unsloth_${t%%:*} "prebuilt $P missing" "$P"; fi
        done
      fi ;;
    *) log "unknown arm $ARM"; exit 2 ;;
  esac
done

"$PY_SUMM" "$SUMM" table --outdir "$RES" --model "$MODEL" > /dev/null && log "wrote $RES/SPEED_TABLE.md"
log "ALL_ARMS_DONE ${ARMS[*]}"
