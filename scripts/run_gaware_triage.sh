#!/usr/bin/env bash
# PPL-only triage for G-aware fair sens_wanda: quant + wikitext PPL@n20, NO downstream (fast, ~8min
# each) to check win-feasibility on PPL (must be ≤ bar 148) before spending 32min on arc+hella.
# fair α=0 ≡ per-row Wanda (F=150); α>0 tilts the shared budget toward high-sensitivity rows.
set -u
cd /workspace/PTQResearch
export CUDA_VISIBLE_DEVICES=MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source env.sh >/dev/null 2>&1
OUT=results/gaware; mkdir -p "$OUT"; NTEST=${NTEST:-20}
SUM="$OUT/triage_ppl.txt"

triage () {  # tag sens_exp fair(0|1) [scope]
  local tag="$1" sexp="$2" fair="${3:-1}" scope="${4:-global}"
  local log="$OUT/triage_${tag}.log"
  local fairflag=""; [ "$fair" = "1" ] && fairflag="--sens-fair"
  echo "[triage] $tag sens_exp=$sexp fair=$fair scope=$scope -> $log"
  python src/nosink.py --mask sens_wanda --mask-scope "$scope" --sens-exp "$sexp" $fairflag \
    --norm col --dense-norm sinkhorn --mode vdense --hold-global --ntest "$NTEST" \
    > "$log" 2>&1
  local r=$(grep RESULT "$log" | tail -1)
  echo "$tag | $r" | tee -a "$SUM"
}

for spec in "$@"; do
  IFS=':' read -r tag sexp fair scope <<< "$spec"
  triage "$tag" "$sexp" "${fair:-1}" "${scope:-global}"
done
echo "TRIAGE_DONE"
