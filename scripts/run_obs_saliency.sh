#!/usr/bin/env bash
# OBS/SparseGPT saliency mask (NON-Sinkhorn), the principled beat-PRISM candidate:
# saliency_ij = W_ij² / [H⁻¹]_jj, reusing the OBS Hessian inverse (no extra calibration).
# per_row scope (no row-scale proxy — sidesteps cell-E's marginal-row_std failure).
# 3-bit / global 0.70 / +v-dense, col norm, dense-norm sinkhorn. Bar = PRISM 38.51/30.68/148.
set -u
cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env.sh >/dev/null 2>&1
OUT=results/mask_norm_decomp; CSV=$OUT/csv; mkdir -p "$CSV"; NTEST=20

run_cell () {  # tag mask scope
  local tag="$1" mask="$2" scope="$3"
  local log="$OUT/cell_${tag}.log" done="$OUT/cell_${tag}.done"
  if [ -f "$done" ]; then echo "[skip] $tag: $(grep RESULT "$log" | tail -1)"; return; fi
  echo "[run] cell=$tag mask=$mask scope=$scope -> $log"
  python src/nosink.py --mask "$mask" --mask-scope "$scope" --norm col --dense-norm sinkhorn \
    --mode vdense --hold-global --ntest "$NTEST" --downstream --downstream-tasks arc_easy,hellaswag \
    --downstream-csv-dir "$CSV" --technique-tag "dec_${tag}" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && grep -q DOWNSTREAM_DONE "$log"; then touch "$done"; echo "[ok] $tag $(grep RESULT "$log" | tail -1)";
  else echo "[FAIL] $tag rc=$rc (see $log)"; fi
}

run_cell I_obssal_global obs_saliency global
run_cell H_obssal_perrow obs_saliency per_row
echo "ALL_OBSSAL_DONE"
