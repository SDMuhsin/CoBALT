#!/bin/bash
# accel4bit ARM C: NVFP4 W4A4 PTQ (llm-compressor 0.13.0 QuantizationModifier scheme=NVFP4)
#                  -> vLLM 0.28.0 native FP4 kernels on SM 12.0.
# Usage: scripts/run_accel4bit_nvfp4.sh <gemma-3-4b|medgemma-27b> [MIG-UUID] [stage]
#   stage in: quant | sanity | eval | speed | kernel | bpw | all (default all)
#   env: SCHEME=NVFP4 (default) | NVFP4A16 (weight-only variant, artifact dir nvfp4a16)
#        VENV=/scratch/root/PTQResearch/env_accel_nvfp4 (own venv, see NOTES.md)
set -o pipefail
MODEL=${1:?model short name}; MIG=${2:-MIG-6ec7b494-8fdc-5531-8226-d8b3ea71838a}; STAGE=${3:-all}
SCHEME=${SCHEME:-NVFP4}
ARM=$(echo "$SCHEME" | tr 'A-Z' 'a-z')          # nvfp4 | nvfp4a16
ROOT=/workspace/PTQResearch
VENV=${VENV:-/scratch/root/PTQResearch/env_accel_nvfp4}
PY=$VENV/bin/python
HUB=/scratch/ckp908/prism_hf/hub
case $MODEL in
  gemma-3-4b)   SRC=$HUB/models--unsloth--gemma-3-4b-it/snapshots/bf46152c47f5dd20b896357cb51abc4c03b8ee8c ;;
  medgemma-27b) SRC=$HUB/models--unsloth--medgemma-27b-text-it/snapshots/b780610baf99c087ba3719a77cf0dacec7261a65 ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
OUT=/scratch/root/PTQResearch/accel4bit_models/$MODEL/$ARM
RES=$ROOT/results/accel4bit/$MODEL/$ARM
mkdir -p "$OUT" "$RES"

# ---- environment ----
unset PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME=/scratch/ckp908/prism_hf HF_HUB_CACHE=/scratch/ckp908/prism_hf/hub
export CUDA_VISIBLE_DEVICES=$MIG TMPDIR=/scratch/root/PTQResearch/tmp
export CUDA_HOME=/scratch/root/PTQResearch/cuda-13
# NOTE: /root (home) is at GPFS quota -> keep ALL regenerable caches on /scratch (vLLM inductor/triton died with Errno 122)
CACHE=/scratch/root/PTQResearch/cache; mkdir -p $CACHE/{vllm,triton,inductor,xdg,nv}
export VLLM_CACHE_ROOT=$CACHE/vllm TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor \
       XDG_CACHE_HOME=$CACHE/xdg CUDA_CACHE_PATH=$CACHE/nv FLASHINFER_WORKSPACE_BASE=$CACHE/xdg
export PATH=/scratch/root/PTQResearch/cuda-13/bin:$VENV/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0      # avoid FlashInfer sampler JIT
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn   # lm_eval touches CUDA before EngineCore starts
INCL="--include_path $ROOT/scripts/accel4bit_lmeval_tasks"   # pubmedqa parquet override (datasets>=4 refuses bigbio script)
mkdir -p $TMPDIR
# ---- engine memory knobs (env-overridable GMU / MNBT). Defaults = vLLM defaults 0.85 / unset for BOTH models.
#      medgemma-27b on a 1g.24gb slice CANNOT host the engine at max_model_len 4096 with bf16 KV: weights 16.62 GiB,
#      gmu 0.85 -> KV -1.37 GiB; gmu 0.95 + max_num_batched_tokens 2048 -> KV 1.85 < 2.19 GiB needed for one 4096 seq;
#      gmu 0.98 -> KV 4.72 GiB but CUDA OOM in warm-up (attempts 1-4 in results/accel4bit/medgemma-27b/nvfp4/).
#      Decision: no kv_cache_dtype=fp8, no max_model_len change -> 27B quality eval runs on the 2g.48gb slice
#      MIG-1d47bdbe at the default config; 27B speed comes from the phase-3 one-slice pass (own 24 GB speed run skipped).
GMU=${GMU:-0.85}; MNBT=${MNBT:-}
LMARGS="pretrained=$OUT,quantization=compressed-tensors,max_model_len=4096,gpu_memory_utilization=$GMU,dtype=bfloat16,enable_prefix_caching=False${MNBT:+,max_num_batched_tokens=$MNBT}"
echo "[run] engine knobs: gpu_memory_utilization=$GMU max_num_batched_tokens=${MNBT:-default}"
echo "[run] model=$MODEL scheme=$SCHEME mig=$MIG stage=$STAGE out=$OUT res=$RES"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null | head -2

run_quant() {
  echo "[run] === quant ($SCHEME) ==="; date
  $PY $ROOT/src/accel4bit_nvfp4_quant.py --model-path "$SRC" --out "$OUT" --scheme "$SCHEME" \
     --num-samples ${NSAMP:-512} --max-seq-len 2048 --seed 42 ${QUANT_EXTRA} 2>&1 | tee "$RES/quant.log"
  echo "[run] quant exit=${PIPESTATUS[0]}"; date
}
run_sanity() {
  echo "[run] === sanity generation + kernel selector log (VLLM_LOGGING_LEVEL=DEBUG) ==="
  VLLM_LOGGING_LEVEL=DEBUG $PY $ROOT/src/accel4bit_nvfp4_sanity.py "$OUT" --gpu-mem $GMU ${MNBT:+--max-num-batched-tokens $MNBT} 2>&1 | tee "$RES/sanity_debug.log"
  local rc=${PIPESTATUS[0]}; echo "[run] sanity exit=$rc"
  echo "[run] kernel selector lines:"; grep -h -i -E "for NVFP4 GEMM|Marlin kernel|weight-only FP4|cutlass|emulation" "$RES/sanity_debug.log" | sort | uniq -c | head -20
  return $rc
}
run_eval() {
  echo "[run] === quality eval (lm_eval vllm backend) ==="; date
  lm_eval --model vllm --model_args "$LMARGS" --tasks wikitext,arc_easy,medqa_4options,pubmedqa $INCL \
     --seed 1234 --batch_size auto --output_path "$RES/lmeval_main" 2>&1 | tee "$RES/eval_main.log"
  lm_eval --model vllm --model_args "$LMARGS" --tasks medmcqa --limit 1000 $INCL \
     --seed 1234 --batch_size auto --output_path "$RES/lmeval_medmcqa" 2>&1 | tee "$RES/eval_medmcqa.log"
  $PY - "$RES" <<'EOF'
import glob, json, os, sys
res = sys.argv[1]
for d in ("lmeval_main", "lmeval_medmcqa"):
    for f in glob.glob(os.path.join(res, d, "**", "results_*.json"), recursive=True):
        j = json.load(open(f))
        for task, r in j["results"].items():
            out = {"task": task, "results": r, "n_samples": j.get("n-samples", {}).get(task),
                   "config": j.get("config"), "source": f}
            json.dump(out, open(os.path.join(res, f"eval_{task}.json"), "w"), indent=1)
            print("WROTE", task, {k: v for k, v in r.items() if "perplexity" in k or k.startswith("acc")})
EOF
  date
}
run_speed() {
  echo "[run] === speed (vllm bench; slice $MIG) ==="; date
  COMMON="--model $OUT --quantization compressed-tensors --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization $GMU --no-enable-prefix-caching${MNBT:+ --max-num-batched-tokens $MNBT}"
  vllm bench latency $COMMON --input-len 512 --output-len 128 --batch-size 1 --num-iters 10 --num-iters-warmup 3 \
       --output-json "$RES/speed_latency_512_128.json" 2>&1 | tee "$RES/speed_latency.log"
  vllm bench latency $COMMON --input-len 512 --output-len 1 --batch-size 1 --num-iters 10 --num-iters-warmup 3 \
       --output-json "$RES/speed_latency_512_1_ttft.json" 2>&1 | tee "$RES/speed_ttft.log"
  vllm bench throughput $COMMON --input-len 512 --output-len 128 --num-prompts 256 --max-num-seqs 32 \
       --output-json "$RES/speed_throughput_256x512_128.json" 2>&1 | tee "$RES/speed_throughput.log"
  date
}
run_kernel() {
  echo "[run] === kernel attribution (torch.profiler, in-process engine) ==="
  $PY $ROOT/src/accel4bit_nvfp4_kernelprobe.py --model "$OUT" --out-json "$RES/kernel_profile.json" --gpu-mem $GMU ${MNBT:+--max-num-batched-tokens $MNBT} 2>&1 | tee "$RES/kernel_profile.log"
}
run_bpw() {
  $PY $ROOT/src/accel4bit_nvfp4_bpw.py "$OUT" --json "$RES/bpw.json"
}
case $STAGE in
  quant) run_quant ;; sanity) run_sanity ;; eval) run_eval ;; speed) run_speed ;; kernel) run_kernel ;; bpw) run_bpw ;;
  all) run_quant && run_bpw && run_sanity && run_eval && run_speed && run_kernel ;;
  post) run_bpw && run_sanity && run_eval && run_speed && run_kernel ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "[run] DONE stage=$STAGE"; date
