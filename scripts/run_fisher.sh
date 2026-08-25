#!/usr/bin/env bash
# Exact-Fisher within-row saliency (fisher_sal). saliency=E[g_i²x_j²]·w_ij² (strict generalization
# of Wanda's E[x_j²]·w_ij²). Needs results/fisher_sal/*.pt (src/compute_fisher_saliency.py).
# MODE=triage → quant+PPL@n20 only (fast). MODE=full → +arc_easy,hellaswag downstream.
# per_row isolates the WITHIN-ROW lever (uniform alloc, vs per-row Wanda F=150/34.64/29.30);
# global adds the (measured-harmful) sensitivity allocation. Bar = PRISM/D: 148/38.51/30.68.
set -u
cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env.sh >/dev/null 2>&1
OUT=results/gaware; CSV=$OUT/csv; mkdir -p "$CSV"; NTEST=${NTEST:-20}; MODE=${MODE:-triage}
FSHRINK=${FSHRINK:-1.0}
SUM="$OUT/fisher_${MODE}.txt"

run () {  # tag scope
  local tag="$1" scope="$2"
  local log="$OUT/fisher_${MODE}_${tag}.log"
  local ds=""; [ "$MODE" = "full" ] && ds="--downstream --downstream-tasks arc_easy,hellaswag --downstream-csv-dir $CSV --technique-tag fisher_${tag}"
  echo "[fisher/$MODE] $tag scope=$scope shrink=$FSHRINK -> $log"
  python src/nosink.py --mask fisher_sal --mask-scope "$scope" --fisher-shrink "$FSHRINK" \
    --norm col --dense-norm sinkhorn --mode vdense --hold-global --ntest "$NTEST" $ds > "$log" 2>&1
  local r=$(grep RESULT "$log" | tail -1)
  echo "$tag | $r" | tee -a "$SUM"
  [ "$MODE" = "full" ] && grep -q DOWNSTREAM_DONE "$log" && echo "  downstream: $(grep -iE 'arc_easy|hellaswag' $log | tail -3)"
}

for spec in "$@"; do
  IFS=':' read -r tag scope <<< "$spec"
  run "$tag" "$scope"
done
echo "FISHER_${MODE}_DONE"
