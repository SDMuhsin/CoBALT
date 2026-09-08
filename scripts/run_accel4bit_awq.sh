#!/usr/bin/env bash
# accel4bit arm D (AWQ W4A16 asym g128, llm-compressor AWQModifier -> vLLM 0.28.0 Marlin).
# usage: scripts/run_accel4bit_awq.sh <gemma-3-4b|medgemma-27b> [MIG-UUID] [stages]
#   stages (comma list, default "quant,gen,eval,speed,bpw"): quant gen eval speed bpw
# Everything is re-runnable; each stage logs to results/accel4bit/<model>/awq/.
set -u
MODEL_SHORT=${1:?model short name}
MIG=${2:-MIG-12daede5-c316-57d7-bd83-a55c417864cd}
STAGES=${3:-quant,gen,eval,speed,bpw}
ROOT=/workspace/PTQResearch
HUB=/scratch/ckp908/prism_hf/hub
case "$MODEL_SHORT" in
  gemma-3-4b)   MODEL_PATH=$HUB/models--unsloth--gemma-3-4b-it/snapshots/bf46152c47f5dd20b896357cb51abc4c03b8ee8c ;;
  medgemma-27b) MODEL_PATH=$HUB/models--unsloth--medgemma-27b-text-it/snapshots/b780610baf99c087ba3719a77cf0dacec7261a65 ;;
  *) echo "unknown model $MODEL_SHORT"; exit 2 ;;
esac
OUT=/scratch/root/PTQResearch/accel4bit_models/$MODEL_SHORT/awq
RES=$ROOT/results/accel4bit/$MODEL_SHORT/awq
mkdir -p "$OUT" "$RES"

# ---- env ----
unset PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HOME=/scratch/ckp908/prism_hf HF_HUB_CACHE=/scratch/ckp908/prism_hf/hub
export CUDA_VISIBLE_DEVICES=$MIG
# NOTE: /root (home) is at GPFS quota -> keep every regenerable cache on /scratch
export TMPDIR=/scratch/root/PTQResearch/tmp/awq   # short path: vLLM ZMQ ipc sockets live in TMPDIR and sun_path is limited to 107 chars (scratchpad path is too long)
export VLLM_CACHE_ROOT=/scratch/root/PTQResearch/cache/vllm TRITON_CACHE_DIR=/scratch/root/PTQResearch/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/root/PTQResearch/cache/inductor XDG_CACHE_HOME=/scratch/root/PTQResearch/cache/xdg
export CUDA_CACHE_PATH=/scratch/root/PTQResearch/cache/nv
mkdir -p "$TMPDIR" $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $XDG_CACHE_HOME $CUDA_CACHE_PATH
export CUDA_HOME=/scratch/root/PTQResearch/cuda-13
export PATH=/scratch/root/PTQResearch/cuda-13/bin:/scratch/root/PTQResearch/env_accel_awq/bin:/scratch/root/PTQResearch/env_accel_vllm/bin:$PATH
export PIP_CONFIG_FILE=/dev/null TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # engine-core fork after CUDA init in parent -> "Cannot re-initialize CUDA in forked subprocess"
PY=/scratch/root/PTQResearch/env_accel_awq/bin/python   # overlay venv: lm_eval 0.4.13 + .pth -> env_accel_vllm site-packages
VLLM="$PY -m vllm.entrypoints.cli.main"
QUANT=compressed-tensors
GPU_UTIL=${GPU_UTIL:-0.85}   # override (e.g. 0.35) when sharing the slice with a running quantization; label it in NOTES
MODEL_ARGS="pretrained=$OUT,quantization=$QUANT,max_model_len=4096,gpu_memory_utilization=$GPU_UTIL,dtype=bfloat16,seed=1234,enable_prefix_caching=False"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
stamp() { date '+%F %T'; }

if has quant; then
  echo "[$(stamp)] QUANT $MODEL_SHORT on $MIG -> $OUT" | tee -a "$RES/quant.log"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a "$RES/quant.log"
  $PY "$ROOT/src/accel4bit_awq_quant.py" --model-path "$MODEL_PATH" --out "$OUT" --offload-device cpu 2>&1 | tee -a "$RES/quant.log"
  echo "[$(stamp)] QUANT_EXIT ${PIPESTATUS[0]}" | tee -a "$RES/quant.log"
fi

if has gen; then
  echo "[$(stamp)] GEN sanity (VLLM_LOGGING_LEVEL=DEBUG)" | tee "$RES/gen.log"
  VLLM_LOGGING_LEVEL=DEBUG $PY "$ROOT/src/accel4bit_awq_gen.py" "$OUT" $QUANT 2>&1 | tee -a "$RES/gen.log"
  echo "[$(stamp)] GEN_EXIT ${PIPESTATUS[0]}" | tee -a "$RES/gen.log"
  grep -iE "marlin|machete|cutlass|scaled_mm|QUANT_METHODS|Using .* kernel|kernel" "$RES/gen.log" | sort | uniq -c | sort -rn | head -20 > "$RES/kernel_dispatch.txt"
fi

if has eval; then
  # PROTOCOL quality eval via the shared driver (loads the model once; eval_<task>.json + eval_summary.json in $RES).
  EVAL_MARGS="max_model_len=4096,gpu_memory_utilization=$GPU_UTIL,dtype=bfloat16,quantization=$QUANT,seed=1234,enable_prefix_caching=False"
  # multimodal 4b checkpoint only: skip the dummy-image profiling run (it crashes in the compiled Gemma3 forward) — text-only 27B needs nothing
  [ "$MODEL_SHORT" = gemma-3-4b ] && EVAL_MARGS="$EVAL_MARGS,limit_mm_per_prompt={\"image\":0}"
  echo "[$(stamp)] EVAL (shared driver) model_args=$EVAL_MARGS" | tee "$RES/eval.log"
  $PY "$ROOT/src/accel4bit_lmeval.py" --backend vllm --pretrained "$OUT" --out_dir "$RES" --batch_size auto --seed 1234 \
      --include_path "$ROOT/scripts/accel4bit_lmeval_tasks" --model_args "$EVAL_MARGS" \
      --tasks wikitext medqa_4options arc_easy pubmedqa medmcqa:1000 2>&1 | tee -a "$RES/eval.log"
  echo "[$(stamp)] EVAL_EXIT ${PIPESTATUS[0]}" | tee -a "$RES/eval.log"
  if [ ! -s "$RES/eval_summary.json" ] && grep -qiE "out of memory|No available memory|not enough memory|free memory .* is less than" "$RES/eval.log"; then
    echo "[$(stamp)] EVAL_RETRY with gpu_memory_utilization=0.92 (memory failure at $GPU_UTIL — DEVIATION, label in NOTES)" | tee -a "$RES/eval.log"
    $PY "$ROOT/src/accel4bit_lmeval.py" --backend vllm --pretrained "$OUT" --out_dir "$RES" --batch_size auto --seed 1234 \
        --include_path "$ROOT/scripts/accel4bit_lmeval_tasks" --model_args "${EVAL_MARGS/gpu_memory_utilization=$GPU_UTIL/gpu_memory_utilization=0.92}" \
        --tasks wikitext medqa_4options arc_easy pubmedqa medmcqa:1000 2>&1 | tee -a "$RES/eval.log"
    echo "[$(stamp)] EVAL_RETRY_EXIT ${PIPESTATUS[0]}" | tee -a "$RES/eval.log"
  fi
fi

if has speed; then
  MEMLOG="$RES/speed_gpumem.csv"; : > "$MEMLOG"
  ( while true; do nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader >> "$MEMLOG"; sleep 2; done ) & MEMPID=$!
  COMMON="--model $OUT --quantization $QUANT --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization $GPU_UTIL --seed 1234"
  [ "$MODEL_SHORT" = gemma-3-4b ] && COMMON="$COMMON --limit-mm-per-prompt {\"image\":0}"
  echo "[$(stamp)] SPEED latency 512->128 bs1 x10" | tee "$RES/speed.log"
  $VLLM bench latency $COMMON --input-len 512 --output-len 128 --batch-size 1 --num-iters 10 --num-iters-warmup 3 \
        --output-json "$RES/speed_latency_512_128.json" 2>&1 | tee -a "$RES/speed.log"
  echo "[$(stamp)] SPEED latency 512->1 bs1 (TTFT / prefill)" | tee -a "$RES/speed.log"
  $VLLM bench latency $COMMON --input-len 512 --output-len 1 --batch-size 1 --num-iters 10 --num-iters-warmup 3 \
        --output-json "$RES/speed_latency_512_1.json" 2>&1 | tee -a "$RES/speed.log"
  echo "[$(stamp)] SPEED throughput 256 prompts 512->128 max-num-seqs 32" | tee -a "$RES/speed.log"
  $VLLM bench throughput $COMMON --input-len 512 --output-len 128 --num-prompts 256 --max-num-seqs 32 \
        --output-json "$RES/speed_throughput_256.json" 2>&1 | tee -a "$RES/speed.log"
  if [ ! -s "$RES/speed_latency_512_128.json" ] && grep -qiE "out of memory|No available memory|not enough memory|free memory .* is less than" "$RES/speed.log"; then
    echo "[$(stamp)] SPEED_RETRY with --gpu-memory-utilization 0.92 (memory failure at $GPU_UTIL — DEVIATION, label in NOTES)" | tee -a "$RES/speed.log"
    COMMON="${COMMON/--gpu-memory-utilization $GPU_UTIL/--gpu-memory-utilization 0.92}"
    $VLLM bench latency $COMMON --input-len 512 --output-len 128 --batch-size 1 --num-iters 10 --num-iters-warmup 3 --output-json "$RES/speed_latency_512_128.json" 2>&1 | tee -a "$RES/speed.log"
    $VLLM bench latency $COMMON --input-len 512 --output-len 1 --batch-size 1 --num-iters 10 --num-iters-warmup 3 --output-json "$RES/speed_latency_512_1.json" 2>&1 | tee -a "$RES/speed.log"
    $VLLM bench throughput $COMMON --input-len 512 --output-len 128 --num-prompts 256 --max-num-seqs 32 --output-json "$RES/speed_throughput_256.json" 2>&1 | tee -a "$RES/speed.log"
  fi
  echo "[$(stamp)] SPEED_EXIT" | tee -a "$RES/speed.log"
  kill $MEMPID 2>/dev/null
  $PY - "$RES" <<'EOF'
import json,sys,os
r=sys.argv[1]
def ld(n):
    p=os.path.join(r,n)
    return json.load(open(p)) if os.path.exists(p) else None
a=ld("speed_latency_512_128.json"); b=ld("speed_latency_512_1.json"); t=ld("speed_throughput_256.json")
out={}
if a and b:
    l128=a["avg_latency"]; l1=b["avg_latency"]
    out.update(latency_512_128_s=l128, ttft_512_s=l1, prefill_tok_s=512/l1, decode_tok_s=127/(l128-l1),
               e2e_tok_s_bs1=128/l128, latency_p50=a.get("percentiles",{}).get("50"), latency_p99=a.get("percentiles",{}).get("99"))
if t: out["throughput"]=t
mem=[int(x.split(",")[1].split()[0]) for x in open(os.path.join(r,"speed_gpumem.csv")) if "," in x]
out["peak_gpu_mem_MiB_nvidia_smi"]=max(mem) if mem else None
out["slice"]=os.environ.get("CUDA_VISIBLE_DEVICES")
json.dump(out,open(os.path.join(r,"speed_summary.json"),"w"),indent=1); print(json.dumps(out,indent=1))
EOF
fi

if has bpw; then
  $PY "$ROOT/src/accel4bit_awq_bpw.py" "$OUT" | tee "$RES/bpw.json"
fi
echo "[$(stamp)] ALL_STAGES_DONE $STAGES"
