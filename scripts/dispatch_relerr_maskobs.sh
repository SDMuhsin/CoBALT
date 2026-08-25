#!/usr/bin/env bash
# Mediator relerr probes for the mask×OBS factorial. Lands one probe per free MIG slice
# (concurrency = #slices), self-skips rows already in relerr.csv. Task-independent (no eval),
# so each probe is just one build_model + a 4-seq forward capture.
set -u
cd "$(dirname "$0")/.." || exit 1
source env.sh 2>/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CSV=results/benchmark_ablation_maskobs/relerr.csv
LOGD=results/benchmark_ablation_maskobs/relerr_logs
mkdir -p "$LOGD"

# 7 MIG slices
SLICES=(
  MIG-bef7a31e-4317-582c-a97d-75e9e429441c
  MIG-6ec7b494-8fdc-5531-8226-d8b3ea71838a
  MIG-475dbed1-782f-5c45-ae0c-1d2507268638
  MIG-12daede5-c316-57d7-bd83-a55c417864cd
  MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0
  MIG-71f8f46a-48d6-5405-b3ab-fe89c9ae420f
  MIG-c450ecb6-fb75-5454-b3b7-2be70cc1a3f8
)

# probe list: "method bits sp beta"
PROBES=()
BANDS_B3=(0.4 0.5 0.6 0.7 0.8)
BANDS_B4=(0.4 0.5 0.6 0.7 0.8)
# core 2x2 (beta 0 and 0.5) over all 10 bands, both methods
for sp in "${BANDS_B3[@]}"; do
  for be in 0 0.5; do
    PROBES+=("cobalt-noobs 3 $sp $be"); PROBES+=("cobalt 3 $sp $be")
  done
done
for sp in "${BANDS_B4[@]}"; do
  for be in 0 0.5; do
    PROBES+=("cobalt-noobs 4 $sp $be"); PROBES+=("cobalt 4 $sp $be")
  done
done
# beta-curve at collapse edge (3b-sp0.7) and healthy cell (4b-sp0.4)
for cell in "3 0.7" "4 0.4"; do
  set -- $cell; b=$1; sp=$2
  for be in 0.3 0.7 1.0; do
    PROBES+=("cobalt-noobs $b $sp $be"); PROBES+=("cobalt $b $sp $be")
  done
done

# helper: is (method,bits,sp,beta) already in CSV?
have() { # method bits sp beta
  awk -F, -v m="$1" -v b="$2" -v s="$(printf '%.2f' "$3")" -v be="$4" \
    'NR>1 && $1==m && $2==b && $3==s && ($4==be || $4+0==be+0){f=1} END{exit f?0:1}' "$CSV" 2>/dev/null
}

launch() { # slice "method bits sp beta"
  local slice=$1; shift; read -r m b sp be <<<"$1"
  local tag="${m}_b${b}_sp${sp}_be${be}"
  CUDA_VISIBLE_DEVICES="$slice" python src/diag_ablation_relerr.py \
    --method "$m" --beta "$be" --bits "$b" --sparsity "$sp" --group-size 128 \
    --csv "$CSV" >"$LOGD/$tag.log" 2>&1
}

echo "[relerr] ${#PROBES[@]} probes queued, ${#SLICES[@]} slices"
declare -A PID2   # slice_idx -> pid
i=0
while [ $i -lt ${#PROBES[@]} ]; do
  for si in "${!SLICES[@]}"; do
    pid="${PID2[$si]:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then continue; fi
    [ $i -ge ${#PROBES[@]} ] && break
    p="${PROBES[$i]}"; read -r m b sp be <<<"$p"
    i=$((i+1))
    if have "$m" "$b" "$sp" "$be"; then echo "[skip] $p (in csv)"; continue; fi
    launch "${SLICES[$si]}" "$p" &
    PID2[$si]=$!
    echo "[launch $i/${#PROBES[@]}] slice$si: $p (pid ${PID2[$si]})"
    sleep 2
  done
  sleep 5
done
wait
echo "RELERR_DONE $(date -u) rows=$(( $(wc -l <"$CSV") - 1 ))"
