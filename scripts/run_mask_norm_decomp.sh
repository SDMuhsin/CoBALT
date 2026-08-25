#!/usr/bin/env bash
# MASK-vs-NORM decomposition (the reset's first measurement).
# 2x2 at 3-bit / global 0.70 / +v-dense (hold-global), ALL through the single
# nosink harness so exactly one factor changes per cell:
#   MASK  in {wanda (|W|.||X||, non-Sinkhorn), inverse_mu (PRISM |W|.||X||/(mu1 mu2))}
#   NORM  in {col (sparse-aware col-std, non-Sinkhorn), sinkhorn_sa (PRISM sparse-aware mu1)}
# CONTROL: dense (sp=0) v_proj uses --dense-norm sinkhorn in EVERY cell -> matches real
# PRISM's dense v_proj EXACTLY (verified matrix-level dequant Δ=0), so the GQA bottleneck
# is held constant and the ONLY variable is the treatment of the PRUNED matrices.
# Cell D (inverse_mu+sinkhorn_sa) == real PRISM bit-for-bit -> its downstream is taken
# from the anchor run (results/mask_norm_decomp/anchor_csv); here we only PPL-check D.
# Each of A/B/C: PPL@n20 + downstream arc_easy,hellaswag (full split). Resumable.
set -u
cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env.sh >/dev/null 2>&1

OUT=results/mask_norm_decomp
CSV=$OUT/csv
mkdir -p "$CSV"
NTEST=20

run_cell () {  # tag mask norm  [downstream=1]
  local tag="$1" mask="$2" norm="$3" ds="${4:-1}"
  local log="$OUT/cell_${tag}.log"
  local done="$OUT/cell_${tag}.done"
  if [ -f "$done" ]; then echo "[skip] $tag already done: $(grep RESULT "$log" | tail -1)"; return; fi
  local dsargs=""
  [ "$ds" = "1" ] && dsargs="--downstream --downstream-tasks arc_easy,hellaswag --downstream-csv-dir $CSV --technique-tag dec_${tag}"
  echo "[run] cell=$tag mask=$mask norm=$norm ds=$ds -> $log"
  python src/nosink.py --mask "$mask" --norm "$norm" --dense-norm sinkhorn \
    --mode vdense --hold-global --ntest "$NTEST" $dsargs > "$log" 2>&1
  local rc=$?
  local ok=1
  [ $rc -ne 0 ] && ok=0
  [ "$ds" = "1" ] && ! grep -q "DOWNSTREAM_DONE" "$log" && ok=0
  [ "$ds" = "0" ] && ! grep -q "RESULT" "$log" && ok=0
  if [ $ok -eq 1 ]; then touch "$done"; echo "[ok] $tag  $(grep RESULT "$log" | tail -1)";
  else echo "[FAIL] $tag rc=$rc (see $log)"; fi
}

# D plumbing check first (PPL only): confirm fixed harness reproduces real PRISM ~148.
run_cell D_invmu_sink_PPLCHK inverse_mu sinkhorn_sa 0
# The 2x2 downstream cells:
run_cell A_wanda_col     wanda      col          1
run_cell B_wanda_sink    wanda      sinkhorn_sa  1
run_cell C_invmu_col     inverse_mu col          1
echo "ALL_CELLS_DONE"
