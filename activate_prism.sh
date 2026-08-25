# Source this file to use the persistent PRISM virtualenv.
#   source activate_prism.sh
#
# This repo lives on a shared Compute-Canada-style HPC inside a *non-persistent*
# Apptainer container. The ./env virtualenv (and HF cache under ./cache) live on
# the persistent workspace mount, so they survive container re-creation as long
# as the base /usr/bin/python3.10 stays at the same path (it does — it ships in
# the cuda.sif image).
#
# The host injects two things that break a clean venv; we neutralize them:
#   * PYTHONPATH      -> CC's /cvmfs site-packages shadow our installs
#   * PIP_CONFIG_FILE -> CC's restricted wheelhouse blocks normal PyPI installs

_PRISM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

unset PYTHONPATH
export PIP_CONFIG_FILE=/dev/null

# Persistent HuggingFace cache on the workspace mount (gitignored via cache/)
export HF_HOME="${_PRISM_ROOT}/cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1

# GPUs. This box (2026-08-03) is 2x NVIDIA A40, NOT MIG-sliced, so the block
# below is a no-op and both cards stay visible. It is kept because the project
# has also run on a MIG-sliced box, where CUDA needs an explicit MIG UUID.
# Override either way by exporting PRISM_MIG or CUDA_VISIBLE_DEVICES first.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ -n "${PRISM_MIG:-}" ]; then
        export CUDA_VISIBLE_DEVICES="${PRISM_MIG}"
    else
        _MIG_UUID="$(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[0-9a-f-]*' | head -1)"
        [ -n "${_MIG_UUID}" ] && export CUDA_VISIBLE_DEVICES="${_MIG_UUID}"
    fi
fi

# shellcheck disable=SC1091
source "${_PRISM_ROOT}/env/bin/activate"

echo "[prism] venv: $(command -v python)"
echo "[prism] HF_HOME=${HF_HOME}"
echo "[prism] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
