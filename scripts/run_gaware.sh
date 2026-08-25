#!/usr/bin/env bash
# G-AWARE (output-Fisher) mask experiments. The task-loss Hessian factorizes (K-FAC) as
# H_act ⊗ G; reconstruction assumes G=I. diag_output_fisher measured G FAR from I on every
# matrix. sens_wanda weights each output ROW of the Wanda saliency by s_i^α (s_i = diag G =
# per-output-channel sensitivity) — the exact water-filling allocation of survivors under the
# diagonal task-loss objective. NON-Sinkhorn, PRISM-orthogonal. Needs results/sensitivity/sens.pt
# (src/compute_sensitivity.py). All at 3-bit / global 0.70 / +v-dense, col norm, dense-norm sinkhorn.
# Bar to beat = PRISM/cell D: arc_easy 38.51 / hellaswag 30.68 / PPL 148.
set -u
cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env.sh >/dev/null 2>&1
OUT=results/gaware; CSV=$OUT/csv; mkdir -p "$CSV"; NTEST=${NTEST:-20}

run_cell () {  # tag mask scope sens_exp fair(0|1)
  local tag="$1" mask="$2" scope="$3" sexp="$4" fair="${5:-0}"
  local log="$OUT/cell_${tag}.log" done="$OUT/cell_${tag}.done"
  local fairflag=""; [ "$fair" = "1" ] && fairflag="--sens-fair"
  if [ -f "$done" ]; then echo "[skip] $tag: $(grep RESULT "$log" | tail -1)"; return; fi
  echo "[run] cell=$tag mask=$mask scope=$scope sens_exp=$sexp fair=$fair -> $log"
  python src/nosink.py --mask "$mask" --mask-scope "$scope" --sens-exp "$sexp" $fairflag --norm col \
    --dense-norm sinkhorn --mode vdense --hold-global --ntest "$NTEST" --downstream \
    --downstream-tasks arc_easy,hellaswag --downstream-csv-dir "$CSV" \
    --technique-tag "gaware_${tag}" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && grep -q DOWNSTREAM_DONE "$log"; then touch "$done"; echo "[ok] $tag $(grep RESULT "$log" | tail -1)";
  else echo "[FAIL] $tag rc=$rc (see $log)"; fi
}

# $@ = list of "tag:mask:scope:sens_exp[:fair]" cells
for spec in "$@"; do
  IFS=':' read -r tag mask scope sexp fair <<< "$spec"
  run_cell "$tag" "$mask" "$scope" "$sexp" "${fair:-0}"
done
echo "ALL_GAWARE_DONE"
