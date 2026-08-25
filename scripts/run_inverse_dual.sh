#!/usr/bin/env bash
# Non-Sinkhorn inverse_dual mask (closed-form μ1μ2 replica via the dual decomposition).
# K=per_row (target: replicate J 38.72/30.18/116), L=global (target: replicate C 38.26/30.68/136).
# dense-norm sinkhorn to ISOLATE the mask replica. Bar=PRISM 38.51/30.68/148.
set -u; cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env.sh >/dev/null 2>&1
OUT=results/mask_norm_decomp; CSV=$OUT/csv; NTEST=20
run_cell () { local tag="$1" scope="$2"; local log="$OUT/cell_${tag}.log" done="$OUT/cell_${tag}.done"
  [ -f "$done" ] && { echo "[skip] $tag"; return; }
  echo "[run] $tag scope=$scope -> $log"
  python src/nosink.py --mask inverse_dual --mask-scope "$scope" --norm col --dense-norm sinkhorn \
    --mode vdense --hold-global --ntest "$NTEST" --downstream --downstream-tasks arc_easy,hellaswag \
    --downstream-csv-dir "$CSV" --technique-tag "dec_${tag}" > "$log" 2>&1
  local rc=$?; { [ $rc -eq 0 ] && grep -q DOWNSTREAM_DONE "$log"; } && { touch "$done"; echo "[ok] $tag $(grep RESULT "$log"|tail -1)"; } || echo "[FAIL] $tag rc=$rc"; }
run_cell K_dual_perrow per_row
run_cell L_dual_global global
echo "ALL_DUAL_DONE"
