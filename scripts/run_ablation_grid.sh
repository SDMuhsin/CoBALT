#!/bin/bash
# Bias-correction ablation grid (Study A 2x2 + Study B transfer grafts).
#
# Mirrors run_c4_downstream_grid.sh: each (technique, precision, sparsity) config
# runs as its own subprocess so one failure never kills the batch; resumable via
# done-markers; shardable across MIG slices; glitch-guarded (rc=126 / clock jumps).
#
# ENV KNOBS:
#   ABL_MODEL     model key (default qwen-0.5b)
#   ABL_DATASET   wikitext2 | c4   (default wikitext2)
#   ABL_TECHS     space-separated technique list (default = full study list)
#   ABL_TAG       output subdir under results/ (default ablation_<model>_<dataset>)
#   ABL_DS        if set (any value), also run the full downstream suite
#   PRISM_MIG     MIG UUID to pin; else auto-pick freest slice
#   DS_SHARD      "idx/total" for parallel sharding (default 0/1)
#   DS_TASKS      downstream task list (default all)
#   DS_LIMIT / DS_GEN_LIMIT  optional downstream subsample caps
#   DS_NO_CODEEXEC  if set, disable HumanEval code execution (default ON)
#   PRISM_SEED    deterministic seed (default 0; forwarded to the runner)
set -u
cd /workspace/PRISM

export HF_HOME=/workspace/PRISM/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
unset PYTHONPATH
export PIP_CONFIG_FILE=/dev/null
export PRISM_SEED="${PRISM_SEED:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${ABL_MODEL:-qwen-0.5b}"
DATASET="${ABL_DATASET:-wikitext2}"
TAG="${ABL_TAG:-ablation_${MODEL}_${DATASET}}"

# Default technique list: Study A 2x2 (+ explicit full-PRISM cell & references) and
# Study B grafts with their matched baselines.
DEFAULT_TECHS="abl-base abl-corr abl-extras abl-full prism wanda-sinq \
sparsegpt sparsegpt-corr slim slim-corr jsq-wo jsq-wo-corr"
TECHS="${ABL_TECHS:-$DEFAULT_TECHS}"

# ---- MIG slice selection ---------------------------------------------------
if [ -n "${PRISM_MIG:-}" ]; then
    MIG="$PRISM_MIG"
    echo "[abl] using PRISM_MIG override: $MIG"
else
    MIG="$(python scripts/pick_free_mig.py)"
    [ -n "$MIG" ] && echo "[abl] auto-picked freest MIG slice: $MIG"
fi
[ -n "${MIG:-}" ] && export CUDA_VISIBLE_DEVICES="$MIG"
echo "[abl] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

PY=/workspace/PRISM/env/bin/python

# ---- Output layout ---------------------------------------------------------
OUT=results/$TAG
MAIN_CSV=$TAG/main.csv                               # relative to results/
DOWNSTREAM_DIR=/workspace/PRISM/results/$TAG/downstream
DONEDIR=$OUT/done
LOGDIR=logs
PROGRESS=$OUT/progress.tsv
mkdir -p "$OUT" "$DOWNSTREAM_DIR" "$DONEDIR" "$LOGDIR"
[ -f "$PROGRESS" ] || printf 'technique\tprecision\tsparsity\tppl\tseconds\tstatus\n' > "$PROGRESS"

# ---- Downstream flag assembly (optional) -----------------------------------
DS_FLAGS=()
if [ -n "${ABL_DS:-}" ]; then
    DS_FLAGS=(--downstream --downstream-tasks "${DS_TASKS:-all}"
              --downstream-csv-dir "$DOWNSTREAM_DIR")
    [ -n "${DS_LIMIT:-}" ]     && DS_FLAGS+=(--downstream-limit "$DS_LIMIT")
    [ -n "${DS_GEN_LIMIT:-}" ] && DS_FLAGS+=(--downstream-gen-limit "$DS_GEN_LIMIT")
    [ -z "${DS_NO_CODEEXEC:-}" ] && DS_FLAGS+=(--downstream-allow-codeexec)
    echo "[abl] downstream ON: ${DS_FLAGS[*]}"
fi

# ---- Sharding --------------------------------------------------------------
_SHARD="${DS_SHARD:-0/1}"
SHARD_IDX="${_SHARD%%/*}"
SHARD_TOTAL="${_SHARD##*/}"
echo "[abl] model=$MODEL dataset=$DATASET tag=$TAG shard=${SHARD_IDX}/${SHARD_TOTAL}"
echo "[abl] techniques: $TECHS"

CFG_N=0; TOTAL=0; DONE=0

run() {  # args: technique precision sparsity
    local tech=$1 prec=$2 sp=$3
    local idx=$CFG_N; CFG_N=$((CFG_N+1))
    [ $((idx % SHARD_TOTAL)) -ne "$SHARD_IDX" ] && return
    local tag="${tech}_${prec}bit_$(python3 -c "print(int($sp*100))")sp"
    TOTAL=$((TOTAL+1))
    if [ -f "$DONEDIR/$tag.done" ]; then echo "[skip] $tag"; DONE=$((DONE+1)); return; fi

    local log="$LOGDIR/${TAG}_${tag}.log"
    local t0=$(date +%s)
    "$PY" benchmarks/benchmark_suite.py \
        --model "$MODEL" --technique "$tech" --precision "$prec" \
        --sparsity "$sp" --dataset "$DATASET" \
        "${DS_FLAGS[@]}" \
        --csv "$MAIN_CSV" > "$log" 2>&1
    local rc=$?
    local t1=$(date +%s)
    local ppl
    ppl=$(grep -oE "Perplexity: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+")
    local status="ok"; [ $rc -ne 0 ] && status="rc=$rc"
    if [ -n "$ppl" ] && [ $rc -eq 0 ]; then touch "$DONEDIR/$tag.done"; DONE=$((DONE+1));
    else [ -z "$ppl" ] && status="${status},noppl"; fi
    [ -z "$ppl" ] && ppl="NA"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tech" "$prec" "$sp" "$ppl" "$((t1-t0))" "$status" >> "$PROGRESS"
    echo "[abl] ${tag} -> ppl=${ppl} (${status}, $((t1-t0))s)"
    if [ "$rc" -eq 126 ] || [ $((t1-t0)) -lt 0 ]; then
        echo "[abl] ENV-GLITCH (rc=$rc, dur=$((t1-t0))s) — aborting shard for supervisor retry"; exit 3
    fi
}

echo "[abl] START $(date)"
SPARS="${ABL_SPARS:-0.05 0.25 0.50}"   # override to focus (e.g. "0.50") for cheaper grids
BITS="${ABL_BITS:-3 4 5}"
for tech in $TECHS; do
    for sp in $SPARS; do
        for prec in $BITS; do
            run "$tech" "$prec" "$sp"
        done
    done
done
echo "[abl] DONE $(date)  ${DONE}/${TOTAL} configs complete (shard ${SHARD_IDX}/${SHARD_TOTAL})"
