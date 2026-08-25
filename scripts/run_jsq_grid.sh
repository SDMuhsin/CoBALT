#!/bin/bash
# JSQ qwen-0.5b grid with per-config editing-strength (clip_h) mini-search (paper-faithful).
# For each (variant, precision, sparsity, dataset): run clip_h in $CLIPS, keep best (lowest) PPL.
# Resumable via done-markers; flock-safe TSV append so shards can run concurrently.
# Shard with env: VARIANTS, DSETS, PRECS, SPARS (and PRISM_MIG before sourcing activate).
cd /workspace/PRISM
source activate_prism.sh >/dev/null 2>&1
OUT=results/jsq
mkdir -p "$OUT/done"
TSV="$OUT/jsq_grid_results.tsv"
[ -f "$TSV" ] || echo -e "variant\tprecision\tsparsity\tdataset\tbest_clip\tbest_ppl\tall_clips" > "$TSV"
CLIPS="${CLIPS:-0 0.005 0.01 0.02}"
VARIANTS="${VARIANTS:-jsq jsq-wo}"
DSETS="${DSETS:-wikitext2 c4}"
PRECS="${PRECS:-3 4 5}"
SPARS="${SPARS:-0.05 0.25 0.5}"
echo "[$(date)] shard start: VARIANTS=$VARIANTS DSETS=$DSETS MIG=$CUDA_VISIBLE_DEVICES"
for variant in $VARIANTS; do
 for dataset in $DSETS; do
  for prec in $PRECS; do
   for sp in $SPARS; do
    marker="$OUT/done/${variant}_${prec}_${sp}_${dataset}.done"
    [ -f "$marker" ] && { echo "[skip] $marker"; continue; }
    best_ppl=""; best_clip=""; all=""
    for clip in $CLIPS; do
      ppl=$(JSQ_CLIPH=$clip timeout 900 python benchmarks/benchmark_suite.py --model qwen-0.5b \
            --technique "$variant" --precision "$prec" --sparsity "$sp" --dataset "$dataset" 2>/dev/null \
            | grep -iE "^Perplexity:" | tail -1 | awk '{print $NF}')
      [ -z "$ppl" ] && ppl="NA"
      all="${all}${clip}:${ppl} "
      if [ "$ppl" != "NA" ] && { [ -z "$best_ppl" ] || awk "BEGIN{exit !($ppl<$best_ppl)}"; }; then
        best_ppl=$ppl; best_clip=$clip
      fi
    done
    if [ -n "$best_ppl" ]; then
      ( flock 9; echo -e "${variant}\t${prec}\t${sp}\t${dataset}\t${best_clip}\t${best_ppl}\t${all}" >> "$TSV" ) 9>>"$TSV.lock"
      touch "$marker"
      echo "[done] $variant p$prec s$sp $dataset -> best_clip=$best_clip best_ppl=$best_ppl  (all: $all)"
    else
      echo "[FAIL] $variant p$prec s$sp $dataset all-NA; no marker (will retry on resume)"
    fi
   done
  done
 done
done
echo "[$(date)] GRID SHARD COMPLETE (DSETS=$DSETS VARIANTS=$VARIANTS)"
