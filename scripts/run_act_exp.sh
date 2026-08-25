#!/usr/bin/env bash
# Test the activation-exponent hypothesis: inverse-μ ≈ |W|·‖X‖^γ (μ1 anti-corr ‖X‖).
# per-row Wanda with γ>1. Targets: per-row Wanda γ=1 (F:34.64/29.30/150) → per-row inverse-μ (J:38.72/30.18/116).
set -u; cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env.sh >/dev/null 2>&1
OUT=results/mask_norm_decomp; CSV=$OUT/csv; NTEST=20
run_cell () { local tag="$1" g="$2"; local log="$OUT/cell_${tag}.log" done="$OUT/cell_${tag}.done"
  [ -f "$done" ] && { echo "[skip] $tag"; return; }
  echo "[run] $tag act_exp=$g -> $log"
  python src/nosink.py --mask wanda --mask-scope per_row --wanda-act-exp "$g" --norm col --dense-norm sinkhorn \
    --mode vdense --hold-global --ntest "$NTEST" --downstream --downstream-tasks arc_easy,hellaswag \
    --downstream-csv-dir "$CSV" --technique-tag "dec_${tag}" > "$log" 2>&1
  local rc=$?; { [ $rc -eq 0 ] && grep -q DOWNSTREAM_DONE "$log"; } && { touch "$done"; echo "[ok] $tag $(grep RESULT "$log"|tail -1)"; } || echo "[FAIL] $tag rc=$rc"; }
run_cell M_perrow_g15 1.5
run_cell N_perrow_g20 2.0
echo "ALL_ACTEXP_DONE"
