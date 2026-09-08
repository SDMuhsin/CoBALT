#!/usr/bin/env bash
# MEASURE-FIRST (novel-mask effort, lever #17 "where does the mask bite"):
# residual-stream relerr propagation as sparsity rises, CoBALT prune stage (balanced+OBS)
# vs matched-OBS baseline (wanda+OBS), on 3 healthy families + gemma control.
# Locates the stress boundary (interior relerr -> 1) and whether CoBALT's mask suppresses it
# MORE than the baseline there. Prune-only relerr = the mechanism's native quantity.
set -u
cd /workspace/PTQResearch
source env.sh >/dev/null 2>&1
OUT=results/maskbite
mkdir -p "$OUT"
TSV="$OUT/relerr_sweep.tsv"
echo -e "model\tsp\tvariant\tmax_relerr\tinterior_max\tfinal_relerr" > "$TSV"
for M in tinyllama qwen-1.5b stablelm-2 gemma-2b; do
  for SP in 0.5 0.6 0.7 0.8; do
    for V in balanced+obs wanda+obs; do
      LOG="$OUT/relerr_${M}_sp${SP}_${V//+/_}.log"
      $VENV/bin/python src/diag_prop_variants.py --model "$M" --sparsity "$SP" --variant "$V" --n-seq 4 > "$LOG" 2>&1
      # interior_max = max relerr over non-final layers (final always blips)
      S=$(grep '^SUMMARY' "$LOG" | tail -1)
      MAXR=$(echo "$S" | sed -n 's/.*max_relerr=\([0-9.eE+-]*\).*/\1/p')
      FINR=$(echo "$S" | sed -n 's/.*final_relerr=\([0-9.eE+-]*\).*/\1/p')
      INT=$($VENV/bin/python - "$LOG" <<'PY'
import sys,re
vals=[]
for ln in open(sys.argv[1]):
    m=re.match(r'\s*(\d+)\s+([0-9.eE+-]+)\s*$', ln)
    if m: vals.append(float(m.group(2)))
print(f"{max(vals[:-1]):.4f}" if len(vals)>1 else (f"{vals[0]:.4f}" if vals else "nan"))
PY
)
      echo -e "${M}\t${SP}\t${V}\t${MAXR:-nan}\t${INT}\t${FINR:-nan}" >> "$TSV"
      echo "[done] $M sp$SP $V -> max=${MAXR} interior=${INT}"
    done
  done
done
echo "ALLDONE_MASKBITE_RELERR"
