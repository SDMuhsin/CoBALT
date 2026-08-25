#!/usr/bin/env bash
# Apples-to-apples: column-BALANCED mask vs PUBLISHED baselines (Wanda+AWQ, Wanda+SINQ, SparseGPT)
# + PRISM (internal ref), at 3-bit, BOTH uniform-0.70 and v-dense. One driver, identical PPL(n40) +
# full arc/hella. Resumable (.done markers). Already have: balanced:vdense, prism:vdense (filled from
# results/gaware). Ordered most-informative-first (balanced:uniform = standard-config headline).
set -u
cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env.sh >/dev/null 2>&1
OUT=results/baseline_compare; CSV=$OUT/csv; mkdir -p "$CSV"; NTEST=${NTEST:-40}
SUM="$OUT/summary.txt"

cell () {  # technique mode
  local tq="$1" mode="$2" tag="${1}_${2}"
  local log="$OUT/${tag}.log" done="$OUT/${tag}.done"
  if [ -f "$done" ]; then echo "[skip] $tag: $(grep RESULT "$log" | tail -1 | grep -oE 'ppl=[0-9.e+]+')"; return; fi
  echo "[run] $tag -> $log"
  python src/baseline_compare.py --technique "$tq" --mode "$mode" --ntest "$NTEST" --downstream \
    --downstream-csv-dir "$CSV" --technique-tag "$tag" > "$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && grep -q DOWNSTREAM_DONE "$log"; then
    touch "$done"
    echo "$tag | $(grep RESULT "$log" | tail -1 | grep -oE 'global_sparsity=[0-9.]+|ppl=[0-9.e+]+' | tr '\n' ' ')" | tee -a "$SUM"
  else
    echo "[FAIL] $tag rc=$rc (see $log)"
  fi
}

for spec in "$@"; do
  IFS=':' read -r tq mode <<< "$spec"
  cell "$tq" "$mode"
done
echo "BASELINE_COMPARE_DONE"
