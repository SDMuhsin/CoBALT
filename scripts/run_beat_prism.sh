#!/usr/bin/env bash
# Beat-PRISM attempts (non-Sinkhorn), run ONLY after cell E validates that the
# inverse_scale mask reproduces cell C (inverse-μ). The MASK is PRISM's lever
# (decomposition verdict), so the win must come from a BETTER non-Sinkhorn mask.
# Motivated by M1 (PRISM's GLOBAL mask starves ~22% of rows) + the secondary
# FINDINGS hypothesis: a PER-ROW threshold guarantees no dead output channel.
# All at 3-bit / global 0.70 / +v-dense (dense-norm sinkhorn, held constant), col norm.
# Bar to beat = PRISM/cell D: arc_easy 38.51 / hellaswag 30.68 / PPL 148.
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

run_cell F_wanda_perrow       wanda         per_row
run_cell G_scale_perrow       inverse_scale per_row
echo "ALL_BEATPRISM_DONE"
