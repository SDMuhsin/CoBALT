# env.sh -- source this, do not run it.
#   source env.sh
#
# Keeps the venv and all heavy caches on /scratch (large, wipeable) and the
# repo (code/docs) on /workspace (small, persistent quota), per
# llmdocs/RTX_SERVER_TUTORIAL.md. Reuses the gated gemma-2b weights + MMLU /
# wikitext datasets already cached under /scratch/ckp908/prism_hf so nothing
# gated has to be re-downloaded.

PROJ=PTQResearch
SCRATCH_HOME=/scratch/root/$PROJ

# Host injects two things that break a clean venv (see activate_prism.sh):
#   * PYTHONPATH      -> CC /cvmfs site-packages shadow our installs
#   * PIP_CONFIG_FILE -> CC restricted wheelhouse blocks normal PyPI installs
unset PYTHONPATH
export PIP_CONFIG_FILE=/dev/null

export VENV="$SCRATCH_HOME/env"

# HuggingFace: reuse the existing shared cache (gemma-2b is gated; it and the
# MMLU/wikitext datasets are already downloaded there). Everything is cached,
# so run offline to avoid any hub auth round-trip for the gated model.
export HF_HOME="/scratch/ckp908/prism_hf"
export HF_HUB_CACHE="$HF_HOME/hub"               # authoritative model/dataset cache
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Other tool caches on scratch
export TORCH_HOME="$SCRATCH_HOME/cache/torch"
export PIP_CACHE_DIR="$SCRATCH_HOME/cache/pip"
export XDG_CACHE_HOME="$SCRATCH_HOME/cache"
export TMPDIR="$SCRATCH_HOME/tmp"

# GPUs on this box are MIG-sliced and shared. Pick the slice on the physical
# GPU with the most free memory (scripts/pick_free_mig.py). Override by
# exporting CUDA_VISIBLE_DEVICES before sourcing.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -x "$VENV/bin/python" ]; then
    _MIG_UUID="$("$VENV/bin/python" "$(dirname "${BASH_SOURCE[0]:-$0}")/scripts/pick_free_mig.py" 2>/dev/null)"
    [ -n "${_MIG_UUID}" ] && export CUDA_VISIBLE_DEVICES="${_MIG_UUID}"
fi

[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"

echo "[ptq] venv:   $(command -v python)"
echo "[ptq] HF_HOME=$HF_HOME (offline=$HF_HUB_OFFLINE)"
echo "[ptq] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
