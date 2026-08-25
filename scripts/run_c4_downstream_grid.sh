#!/bin/bash
# Full C4 + downstream-eval grid for Qwen-0.5B.
#
# Mirrors cache/repro/run_qwen_grid.sh: each (technique, precision, sparsity)
# config runs as its own subprocess so a single failure never kills the batch.
# Every run does a C4 perplexity eval AND the full downstream-task suite
# (hellaswag, arc_easy, arc_challenge, lambada, mmlu, math, mrr, humaneval).
#
# Outputs land under results/c4_downstream/:
#   main.csv                     <- one PPL row per config (--csv c4_downstream/main.csv)
#   downstream/<task>.csv        <- one row per (config, task)
#   done/<tag>.done              <- resume markers (touched only on success)
#   progress.tsv                 <- live TSV progress log
# Per-config stdout/stderr go to logs/c4_<tag>.log in the repo root.
#
# RESUMABLE: re-running skips any config whose done-marker exists. Delete a
# marker (or the whole done/ dir) to force a re-run.
#
# ENV KNOBS:
#   PRISM_MIG       MIG UUID to pin to. If unset, auto-picks the freest slice
#                   via scripts/pick_free_mig.py.
#   DS_LIMIT        Subsample N examples per downstream task (empty = full split).
#   DS_GEN_LIMIT    Separate cap for generative tasks math/humaneval (empty = no cap).
#   DS_NO_CODEEXEC  If set (any value), DISABLE HumanEval code execution.
#                   Default (unset) = code execution ON.
#
# set -u but NOT -e: we never want one bad config to abort the whole grid.
set -u

cd /workspace/PRISM

export HF_HOME=/workspace/PRISM/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
unset PYTHONPATH
export PIP_CONFIG_FILE=/dev/null

# ---- MIG slice selection ---------------------------------------------------
# Prefer an explicit PRISM_MIG; otherwise auto-pick the freest slice.
if [ -n "${PRISM_MIG:-}" ]; then
    MIG="$PRISM_MIG"
    echo "[c4grid] using PRISM_MIG override: $MIG"
else
    MIG="$(python scripts/pick_free_mig.py)"
    if [ -n "$MIG" ]; then
        echo "[c4grid] auto-picked freest MIG slice: $MIG"
    else
        echo "[c4grid] no MIG slice found; leaving CUDA_VISIBLE_DEVICES as-is"
    fi
fi
[ -n "$MIG" ] && export CUDA_VISIBLE_DEVICES="$MIG"
echo "[c4grid] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

PY=/workspace/PRISM/env/bin/python

# ---- Output layout ---------------------------------------------------------
OUT=results/c4_downstream
MAIN_CSV=c4_downstream/main.csv                                  # relative to results/
DOWNSTREAM_DIR=/workspace/PRISM/results/c4_downstream/downstream # absolute
DONEDIR=$OUT/done
LOGDIR=logs
PROGRESS=$OUT/progress.tsv

mkdir -p "$OUT" "$DOWNSTREAM_DIR" "$DONEDIR" "$LOGDIR"

# Write the progress header only once (keep prior rows on resume).
if [ ! -f "$PROGRESS" ]; then
    printf 'technique\tprecision\tsparsity\tppl\tseconds\tstatus\n' > "$PROGRESS"
fi

# ---- Downstream flag assembly ----------------------------------------------
# Always run the full downstream suite. Limit knobs are optional (empty = full).
DS_FLAGS=(--downstream --downstream-tasks "${DS_TASKS:-all}")
[ -n "${DS_LIMIT:-}" ]     && DS_FLAGS+=(--downstream-limit "$DS_LIMIT")
[ -n "${DS_GEN_LIMIT:-}" ] && DS_FLAGS+=(--downstream-gen-limit "$DS_GEN_LIMIT")
# HumanEval code execution defaults ON; DS_NO_CODEEXEC (set) turns it off.
if [ -z "${DS_NO_CODEEXEC:-}" ]; then
    DS_FLAGS+=(--downstream-allow-codeexec)
fi

echo "[c4grid] downstream flags: ${DS_FLAGS[*]}"

# ---- Sharding (parallel instances across GPUs) -----------------------------
# DS_SHARD="idx/total" makes this instance run only configs whose 0-based index
# in the matrix satisfies (index % total == idx). Default "0/1" = run all.
_SHARD="${DS_SHARD:-0/1}"
SHARD_IDX="${_SHARD%%/*}"
SHARD_TOTAL="${_SHARD##*/}"
echo "[c4grid] shard ${SHARD_IDX}/${SHARD_TOTAL}  tasks=${DS_TASKS:-all}"

CFG_N=0
TOTAL=0
DONE=0

run() {  # args: technique precision sparsity
    local tech=$1 prec=$2 sp=$3
    # Sharding: skip configs not owned by this instance (deterministic order).
    local idx=$CFG_N
    CFG_N=$((CFG_N+1))
    if [ $((idx % SHARD_TOTAL)) -ne "$SHARD_IDX" ]; then
        return
    fi
    local tag="${tech}_${prec}bit_$(python3 -c "print(int($sp*100))")sp"
    TOTAL=$((TOTAL+1))

    if [ -f "$DONEDIR/$tag.done" ]; then
        echo "[skip] $tag"
        DONE=$((DONE+1))
        return
    fi

    local log="$LOGDIR/c4_${tag}.log"
    local t0=$(date +%s)
    "$PY" benchmarks/benchmark_suite.py \
        --model qwen-0.5b --technique "$tech" --precision "$prec" \
        --sparsity "$sp" --dataset c4 \
        "${DS_FLAGS[@]}" \
        --downstream-csv-dir "$DOWNSTREAM_DIR" \
        --csv "$MAIN_CSV" > "$log" 2>&1
    local rc=$?
    local t1=$(date +%s)

    local ppl
    ppl=$(grep -oE "Perplexity: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+")

    local status="ok"
    [ $rc -ne 0 ] && status="rc=$rc"
    if [ -n "$ppl" ] && [ $rc -eq 0 ]; then
        touch "$DONEDIR/$tag.done"
        DONE=$((DONE+1))
    else
        [ -z "$ppl" ] && status="${status},noppl"
    fi
    [ -z "$ppl" ] && ppl="NA"

    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$tech" "$prec" "$sp" "$ppl" "$((t1-t0))" "$status" >> "$PROGRESS"
    echo "[c4grid] ${tag} -> ppl=${ppl} (${status}, $((t1-t0))s)"

    # Glitch guard: rc=126 ("cannot execute") or a negative wall-time (clock jump)
    # means the box/container is broken. Abort the whole shard immediately instead
    # of burning through every remaining config as an instant failure; the
    # supervisor will relaunch once the box recovers. (Real code bugs give rc=1,
    # not 126, so this does not mask genuine per-config failures.)
    if [ "$rc" -eq 126 ] || [ $((t1-t0)) -lt 0 ]; then
        echo "[c4grid] ENV-GLITCH detected (rc=$rc, dur=$((t1-t0))s) — aborting shard for supervisor retry"
        exit 3
    fi
}

echo "[c4grid] START $(date)"

# ---- The 25-config matrix (C4 only) ----------------------------------------
# PRISM (the contribution) first.
for sp in 0.05 0.25 0.50; do
    for prec in 3 4 5; do run prism "$prec" "$sp"; done
done
# SparseGPT baseline.
for sp in 0.05 0.25 0.50; do
    for prec in 3 4 5; do run sparsegpt "$prec" "$sp"; done
done
# Wanda (FP16 weights, pruning only) - precision arg ignored, use 4.
for sp in 0.05 0.25 0.50; do run wanda 4 "$sp"; done
# SINQ (dense quant, no sparsity).
for prec in 3 4 5; do run sinq "$prec" 0.0; done
# FP16 reference.
run fp16 4 0.0

echo "[c4grid] DONE $(date)"
echo "[c4grid] summary: ${DONE}/${TOTAL} configs complete"
echo "[c4grid] main csv:        results/$MAIN_CSV"
echo "[c4grid] downstream csvs:  $DOWNSTREAM_DIR/"
echo "[c4grid] progress tsv:     $PROGRESS"
