#!/bin/bash
# lm_eval (accel4bit PROTOCOL) on a CoBALT fake-quantised HF checkpoint.
#   scripts/run_cobaltkernel_smoke_eval.sh <model-dir> <out-dir> [MIG] [tasks...]
set -u
M=${1:?model dir}; OUT=${2:?out dir}; MIG=${3:-MIG-c450ecb6-fb75-5454-b3b7-2be70cc1a3f8}
shift 3 2>/dev/null || shift $#
TASKS=("$@"); [ ${#TASKS[@]} -eq 0 ] && TASKS=(wikitext arc_easy)
ROOT=/workspace/PTQResearch
ENV=/scratch/root/PTQResearch/env_accel_ref
unset PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HOME=/scratch/ckp908/prism_hf HF_HUB_CACHE=/scratch/ckp908/prism_hf/hub
export TMPDIR=/scratch/root/PTQResearch/tmp PIP_CONFIG_FILE=/dev/null
export CUDA_VISIBLE_DEVICES=$MIG CUDA_HOME=/scratch/root/PTQResearch/cuda-13
export PATH=/scratch/root/PTQResearch/cuda-13/bin:$ENV/bin:$PATH
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
C=/scratch/root/PTQResearch/cache; mkdir -p $C/vllm $C/triton $C/inductor $C/xdg $C/nv
export VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg CUDA_CACHE_PATH=$C/nv
mkdir -p "$OUT"
exec $ENV/bin/python $ROOT/src/accel4bit_lmeval.py --backend vllm --pretrained "$M" --out_dir "$OUT" \
  --model_args max_model_len=4096,gpu_memory_utilization=0.85,dtype=bfloat16,enable_prefix_caching=False,seed=1234 \
  --batch_size auto --seed 1234 --include_path $ROOT/scripts/accel4bit_lmeval_tasks --tasks "${TASKS[@]}"
