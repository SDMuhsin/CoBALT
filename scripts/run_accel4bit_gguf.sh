#!/bin/bash
# accel4bit ARM B (gguf): llama.cpp imatrix Q4_K_M (own quantization from bf16) -> llama.cpp CUDA MMQ kernels.
#
# usage: scripts/run_accel4bit_gguf.sh <gemma-3-4b|medgemma-27b> [step ...] [MIG-UUID via env MIG]
# steps: calib wiki convert imatrix quantize gen ppl eval bench batched prebuilt(27b only) all
#        eval/ppl/bench/batched take TAG=own|unsloth_q4km|unsloth_udq4kxl (default own) to pick the GGUF.
# env overrides: MIG (slice UUID), IMATRIX_NGL, IMATRIX_SRC=bf16|q8, PORT, TASKS, LIMIT_MEDMCQA, NTHREADS
set -u
MODEL=${1:?model short name}; shift
STEPS=${*:-all}

# ---------------------------------------------------------------- environment (never touch the project venv)
unset PYTHONPATH LIBRARY_PATH
export PIP_CONFIG_FILE=/dev/null TMPDIR=/scratch/root/PTQResearch/tmp
# NOTE: /root (home GPFS) is at quota -> keep all caches/tmp on /scratch
export CUDA_CACHE_PATH=/scratch/root/PTQResearch/cache/nv XDG_CACHE_HOME=/scratch/root/PTQResearch/cache/xdg
export TRITON_CACHE_DIR=/scratch/root/PTQResearch/cache/triton TORCHINDUCTOR_CACHE_DIR=/scratch/root/PTQResearch/cache/inductor VLLM_CACHE_ROOT=/scratch/root/PTQResearch/cache/vllm
mkdir -p $CUDA_CACHE_PATH $XDG_CACHE_HOME $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMPDIR
export HF_HOME=/scratch/ckp908/prism_hf HF_HUB_CACHE=/scratch/ckp908/prism_hf/hub
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
CUDA13=/scratch/root/PTQResearch/cuda-13
export LD_LIBRARY_PATH=$CUDA13/lib:/.singularity.d/libs
export CUDA_VISIBLE_DEVICES=${MIG:-MIG-475dbed1-782f-5c45-ae0c-1d2507268638}
V=/scratch/root/PTQResearch/env_accel_llamacpp
PY=$V/bin/python
ROOT=/workspace/PTQResearch
LC=$ROOT/temp/llama.cpp
BIN=$LC/build/bin                 # default build: MMQ kernels (GGML_CUDA_FORCE_CUBLAS=OFF)
BIN_CUBLAS=$LC/build-cublas/bin   # counterfactual build: -DGGML_CUDA_FORCE_CUBLAS=ON (proof that MMQ is dispatched)
NTHREADS=${NTHREADS:-16}
PORT=${PORT:-18475}
TAG=${TAG:-own}
TASKS=${TASKS:-"wikitext arc_easy medqa_4options pubmedqa medmcqa"}
LIMIT_MEDMCQA=${LIMIT_MEDMCQA:-1000}

case $MODEL in
  gemma-3-4b)
    HF=/scratch/ckp908/prism_hf/hub/models--unsloth--gemma-3-4b-it/snapshots/bf46152c47f5dd20b896357cb51abc4c03b8ee8c
    NAME=gemma-3-4b-it; IMATRIX_NGL=${IMATRIX_NGL:-99}; IMATRIX_SRC=${IMATRIX_SRC:-bf16} ;;
  medgemma-27b)
    HF=/scratch/ckp908/prism_hf/hub/models--unsloth--medgemma-27b-text-it/snapshots/b780610baf99c087ba3719a77cf0dacec7261a65
    NAME=medgemma-27b-text-it; IMATRIX_NGL=${IMATRIX_NGL:-22}; IMATRIX_SRC=${IMATRIX_SRC:-bf16} ;;
  *) echo "unknown model $MODEL"; exit 2 ;;
esac
TOKENIZER_HF=/scratch/ckp908/prism_hf/hub/models--unsloth--medgemma-27b-text-it/snapshots/b780610baf99c087ba3719a77cf0dacec7261a65
MD=/scratch/root/PTQResearch/accel4bit_models/$MODEL/gguf
RES=$ROOT/results/accel4bit/$MODEL/gguf
CALIB=$ROOT/results/accel4bit/calib_ultrachat_512x2048.txt
WIKI=$ROOT/results/accel4bit/wiki.test.raw
BF16=$MD/$NAME-bf16.gguf
Q8=$MD/$NAME-Q8_0.gguf
IMATRIX=$MD/$NAME-imatrix.gguf
Q4=$MD/$NAME-Q4_K_M.gguf
mkdir -p $MD $RES

gguf_for_tag() {
  case $1 in
    own) echo $Q4 ;;
    unsloth_q4km)    echo /scratch/root/PTQResearch/accel4bit_models/medgemma-27b/gguf/unsloth_prebuilt/medgemma-27b-text-it-Q4_K_M.gguf ;;
    unsloth_udq4kxl) echo /scratch/root/PTQResearch/accel4bit_models/medgemma-27b/gguf/unsloth_prebuilt/medgemma-27b-text-it-UD-Q4_K_XL.gguf ;;
    *) echo "bad TAG $1" >&2; exit 2 ;;
  esac
}
stamp() { echo "[$(date '+%F %T')] $*"; }
# per-process GPU memory of THIS arm's processes (nvidia-smi --query-gpu reports the whole parent GPU under MIG, i.e. other arms too)
gpu_used_mib() { nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | grep -E 'temp/llama.cpp|env_accel_llamacpp' | awk -F', ' '{s+=$2} END {print s+0}'; }
# sample peak GPU memory of the slice while a command runs: peak_run <outfile> <cmd...>
peak_run() {
  local out=$1; shift
  local peak=0
  ( while true; do m=$(gpu_used_mib); [ -n "$m" ] && echo $m; sleep 1; done ) > $out.samples &
  local sp=$!
  "$@"; local rc=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  peak=$(sort -n $out.samples | tail -1); echo "peak_gpu_mem_mib=$peak" | tee $out; rm -f $out.samples
  return $rc
}

step_calib() {
  stamp calib
  $PY $ROOT/src/accel4bit_dump_calib.py --tokenizer $TOKENIZER_HF --out $CALIB 2>&1 | tee -a $RES/calib.log
}

step_wiki() {
  stamp wiki
  [ -f $WIKI ] && { echo "$WIKI exists"; return 0; }
  ( cd $(dirname $WIKI) && curl -sSL -o wikitext-2-raw-v1.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip \
      && unzip -o -q wikitext-2-raw-v1.zip && mv wikitext-2-raw/wiki.test.raw . && rm -rf wikitext-2-raw wikitext-2-raw-v1.zip )
  md5sum $WIKI
}

step_convert() {
  stamp convert
  [ -f $BF16 ] && { echo "$BF16 exists -> skip"; return 0; }
  $PY $LC/convert_hf_to_gguf.py $HF --outtype bf16 --outfile $BF16 2>&1 | tee $RES/convert.log | tail -5
  ls -la $BF16
}

step_imatrix() {
  stamp imatrix "(src=$IMATRIX_SRC ngl=$IMATRIX_NGL)"
  local SRC=$BF16
  if [ $IMATRIX_SRC = q8 ]; then
    [ -f $Q8 ] || $BIN/llama-quantize $BF16 $Q8 Q8_0 $NTHREADS 2>&1 | tee $RES/quant_q8.log | tail -3
    SRC=$Q8
  fi
  peak_run $RES/imatrix.peakmem $BIN/llama-imatrix -m $SRC -f $CALIB -o $IMATRIX -c 512 --parse-special -ngl $IMATRIX_NGL -t $NTHREADS \
      --output-frequency 100 --save-frequency 0 2>&1 | tee $RES/imatrix.log | grep -vE '^\[[0-9]+\]' | tail -20
  grep -E 'chunks|compute_imatrix: tokenizing|tokens|Final estimate|stored collected' $RES/imatrix.log | tail -8
}

step_quantize() {
  stamp quantize
  $BIN/llama-quantize --imatrix $IMATRIX $BF16 $Q4 Q4_K_M $NTHREADS 2>&1 | tee $RES/quant.log | tail -12
  ls -la $Q4
  $PY - "$Q4" "$RES/quant_summary.json" <<'PY'
import sys, json, gguf
p, out = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(p)
by_type, n_lin, bits_lin = {}, 0, 0
for t in r.tensors:
    n = int(t.n_elements); ty = t.tensor_type.name
    by_type.setdefault(ty, [0, 0]); by_type[ty][0] += 1; by_type[ty][1] += n
    if t.name.endswith('.weight') and ('attn_' in t.name or 'ffn_' in t.name) and 'norm' not in t.name:
        n_lin += n; bits_lin += int(t.n_bytes) * 8
import os
tot = sum(v[1] for v in by_type.values()); size = os.path.getsize(p)
d = dict(file=p, size_bytes=size, n_params_all=tot, bpw_all_from_file=size * 8 / tot,
         n_params_linear=n_lin, bpw_linear=bits_lin / max(n_lin, 1), types={k: dict(n_tensors=v[0], n_elements=v[1]) for k, v in by_type.items()})
json.dump(d, open(out, 'w'), indent=2); print(json.dumps(d, indent=2))
PY
}

start_server_native() {  # llama-server from OUR build (used for the generation check only; no prompt logprobs -> not usable for lm_eval)
  $BIN/llama-server -m $1 -ngl 99 --ctx-size 4096 -np 4 --port $PORT --host 127.0.0.1 > $2 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 600); do curl -s http://127.0.0.1:$PORT/health | grep -q '"ok"' && return 0; kill -0 $SRV_PID 2>/dev/null || { echo "server died"; tail -20 $2; return 1; }; sleep 1; done
  return 1
}
start_server_lcp() {   # llama-cpp-python OpenAI-compatible server: supports echo+logprobs (prompt logprobs) needed by lm_eval loglikelihood
  $PY -m llama_cpp.server --model $1 --n_gpu_layers 99 --n_ctx 4096 --n_batch 512 --logits_all true --n_threads $NTHREADS \
      --host 127.0.0.1 --port $PORT --model_alias $NAME --verbose false > $2 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 900); do curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null | grep -q 200 && return 0; kill -0 $SRV_PID 2>/dev/null || { echo "server died"; tail -20 $2; return 1; }; sleep 1; done
  return 1
}
stop_server() { kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; }

step_gen() {
  local G=$(gguf_for_tag $TAG); stamp gen $TAG $G
  start_server_native $G $RES/server_gen_$TAG.log || return 1
  for q in "Name three symptoms of diabetes." "What is the first-line treatment for uncomplicated hypertension in a 55-year-old?"; do
    echo "### PROMPT: $q"
    curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":96,\"temperature\":0}" | $PY -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
  done 2>&1 | tee $RES/gen_$TAG.txt
  stop_server
}

step_ppl() {   # native llama-perplexity (labelled separately from the lm_eval wikitext numbers)
  local G=$(gguf_for_tag $TAG); stamp ppl $TAG
  peak_run $RES/ppl_native_$TAG.peakmem $BIN/llama-perplexity -m $G -f $WIKI -c 2048 -ngl 99 -t $NTHREADS 2>&1 | tee $RES/ppl_native_$TAG.log | grep -E 'Final estimate|error'
}

step_eval() {   # lm_eval 0.4.13 quality eval, in-process llama-cpp-python (see src/accel4bit_lmeval_gguf.py header for why not llama-server)
  local G=$(gguf_for_tag $TAG); stamp eval $TAG $G
  peak_run $RES/eval_$TAG.peakmem $PY $ROOT/src/accel4bit_lmeval_gguf.py --model_path $G --tasks $TASKS --output_dir $RES --tag $TAG \
      --limit_medmcqa $LIMIT_MEDMCQA --include_path $ROOT/scripts/accel4bit_lmeval_tasks --n_ctx 4096 --n_threads $NTHREADS --seed 1234 \
      2>&1 | tee $RES/eval_$TAG.log | grep -E 'RESULT|SUMMARY|Error|error|Traceback' | tail -12
}
step_eval_server_fallback() {   # documented fallback (NOT used for reported numbers): llama-cpp-python server + lm_eval local-completions; ~12 s/request
  local G=$(gguf_for_tag $TAG); stamp eval-server-fallback $TAG $G
  start_server_lcp $G $RES/server_eval_$TAG.log || return 1
  for task in $TASKS; do
    local lim=""; [ $task = medmcqa ] && lim="--limit $LIMIT_MEDMCQA"
    $V/bin/lm_eval --model local-completions --tasks $task --seed 1234 --batch_size 1 $lim --include_path $ROOT/scripts/accel4bit_lmeval_tasks \
      --model_args base_url=http://127.0.0.1:$PORT/v1/completions,model=$NAME,tokenized_requests=False,tokenizer_backend=huggingface,tokenizer=$TOKENIZER_HF,add_bos_token=True,num_concurrent=4,max_retries=3,max_length=4096 \
      --output_path $RES/evalsrv_${task}_$TAG.json --log_samples 2>&1 | tee $RES/evalsrv_${task}_$TAG.log | grep -E '^\||Error|error' | tail -8
  done
  stop_server
}

step_bench() {   # single-stream: pp512 / tg128, MMQ build vs forced-cuBLAS build
  local G=$(gguf_for_tag $TAG); stamp bench $TAG
  for b in mmq cublas; do
    local B=$BIN; [ $b = cublas ] && B=$BIN_CUBLAS
    peak_run $RES/speed_bench_${b}_$TAG.peakmem $B/llama-bench -m $G -ngl 99 -p 512 -n 128 -r 5 -o json 2> $RES/speed_bench_${b}_$TAG.stderr > $RES/speed_bench_${b}_$TAG.json
    $B/llama-bench -m $G -ngl 99 -p 512 -n 128 -r 5 2>/dev/null | grep -E '^\| (gemma|model)' | tee $RES/speed_bench_${b}_$TAG.md
  done
}

step_batched() {
  local G=$(gguf_for_tag $TAG); stamp batched $TAG
  peak_run $RES/speed_batched_$TAG.peakmem $BIN/llama-batched-bench -m $G -ngl 99 -npp 512 -ntg 128 -npl 1,8,32 -c 32768 -b 4096 -ub 512 -t $NTHREADS 2>&1 | tee $RES/speed_batched_$TAG.log | grep -E '^\|'
}

step_prebuilt() {   # second data point: unsloth prebuilt GGUFs (label "unsloth prebuilt")
  for t in unsloth_q4km unsloth_udq4kxl; do
    TAG=$t step_bench; TAG=$t step_batched; TAG=$t step_ppl; TAG=$t step_eval
  done
}

for s in $STEPS; do
  case $s in
    all) step_calib; step_wiki; step_convert; step_imatrix; step_quantize; step_gen; step_ppl; step_bench; step_batched; step_eval ;;
    calib|wiki|convert|imatrix|quantize|gen|ppl|eval|bench|batched|prebuilt) step_$s ;;
    *) echo "unknown step $s"; exit 2 ;;
  esac
done
stamp done
