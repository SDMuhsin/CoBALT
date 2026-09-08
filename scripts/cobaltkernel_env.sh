# scripts/cobaltkernel_env.sh -- source this, do not run it.
#   source scripts/cobaltkernel_env.sh [MIG_UUID | 2g | 1g | auto]
#
# Sets up the CUDA-13 / torch-2.13 build environment for the cobaltkernel
# custom CUDA extension work (SM 12.0 / sm_120a, Blackwell RTX PRO 6000, MIG).
# Everything heavy lives on /scratch (home has a tight quota).

PROJ=PTQResearch
SCRATCH_HOME=/scratch/root/$PROJ
REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

unset PYTHONPATH
export PIP_CONFIG_FILE=/dev/null

# ---- CUDA 13.0 toolkit (nvcc 13.0.88) --------------------------------------
export CUDA_HOME="$SCRATCH_HOME/cuda-13"
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# ---- venv: main repo env, torch 2.13.0+cu130 (arch list includes sm_120) ----
export VENV="$SCRATCH_HOME/env"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

# ---- caches OFF the home quota ---------------------------------------------
export TORCH_EXTENSIONS_DIR="$SCRATCH_HOME/cache/torch_ext"
export TORCH_HOME="$SCRATCH_HOME/cache/torch"
export PIP_CACHE_DIR="$SCRATCH_HOME/cache/pip"
export XDG_CACHE_HOME="$SCRATCH_HOME/cache"
export TMPDIR="$SCRATCH_HOME/tmp"
export CUDA_CACHE_PATH="$SCRATCH_HOME/cache/nv"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$CUDA_CACHE_PATH"

# ---- HF cache (offline; gemma is gated but already cached) ------------------
export HF_HOME="/scratch/ckp908/prism_hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- target arch ------------------------------------------------------------
# sm_120a = Blackwell consumer/pro arch-specific target (needed for the *a*
# tensor-core / fp8 mma variants). TORCH_CUDA_ARCH_LIST drives torch's
# cpp_extension gencode; "12.0a" maps to compute_120a/sm_120a.
export TORCH_CUDA_ARCH_LIST="12.0a"
export COBALTKERNEL_NVCC_ARCH="-gencode arch=compute_120a,code=sm_120a"

# ---- MIG slice selection ----------------------------------------------------
# 2g.48gb slice used for ALL cross-arm speed numbers (accel4bit protocol):
export COBALT_MIG_2G=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
# a 1g.24gb slice on the other physical GPU (for cheap smoke tests):
export COBALT_MIG_1G=MIG-bef7a31e-4317-582c-a97d-75e9e429441c

_sel="${1:-auto}"
case "$_sel" in
  2g)   export CUDA_VISIBLE_DEVICES="$COBALT_MIG_2G" ;;
  1g)   export CUDA_VISIBLE_DEVICES="$COBALT_MIG_1G" ;;
  auto) if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -x "$VENV/bin/python" ]; then
            _u="$("$VENV/bin/python" "$REPO/scripts/pick_free_mig.py" 2>/dev/null)"
            [ -n "$_u" ] && export CUDA_VISIBLE_DEVICES="$_u"
        fi ;;
  MIG-*) export CUDA_VISIBLE_DEVICES="$_sel" ;;
  *)    echo "[cobaltkernel] unknown slice selector '$_sel' (use 2g|1g|auto|MIG-uuid)" ;;
esac

echo "[cobaltkernel] python : $(command -v python)  torch $(python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
echo "[cobaltkernel] nvcc   : $(command -v nvcc)  $(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | tail -1)"
echo "[cobaltkernel] CUDA_HOME=$CUDA_HOME"
echo "[cobaltkernel] TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "[cobaltkernel] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "[cobaltkernel] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
