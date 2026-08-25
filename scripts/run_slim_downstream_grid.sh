#!/bin/bash
# SLiM-LoRA C4 + downstream-eval grid for Qwen-0.5B (adds `slim` to the existing
# results/c4_downstream/ comparison alongside prism/sparsegpt/wanda/sinq/fp16).
#
# Same methodology as scripts/run_c4_downstream_grid.sh (C4 PPL + full-split
# downstream suite, HumanEval code-exec on), but only the 9 SLiM configs
# (precision {3,4,5} x sparsity {0.05,0.25,0.50}). Writes into the SAME output
# dirs/CSVs so scripts/build_comparison_table.py picks the slim rows up.
#
# RESUMABLE via done/<tag>.done markers. ENV KNOBS: PRISM_MIG, DS_TASKS, DS_LIMIT,
# DS_GEN_LIMIT, DS_NO_CODEEXEC, DS_SHARD="idx/total" (see run_c4_downstream_grid.sh).
set -u
cd /workspace/PRISM

export HF_HOME=/workspace/PRISM/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
unset PYTHONPATH
export PIP_CONFIG_FILE=/dev/null

if [ -n "${PRISM_MIG:-}" ]; then
    MIG="$PRISM_MIG"; echo "[slimgrid] using PRISM_MIG override: $MIG"
else
    MIG="$(python scripts/pick_free_mig.py)"
    [ -n "$MIG" ] && echo "[slimgrid] auto-picked freest MIG slice: $MIG" \
                  || echo "[slimgrid] no MIG slice found; leaving CUDA_VISIBLE_DEVICES as-is"
fi
[ -n "$MIG" ] && export CUDA_VISIBLE_DEVICES="$MIG"
echo "[slimgrid] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

PY=/workspace/PRISM/env/bin/python
OUT=results/c4_downstream
MAIN_CSV=c4_downstream/main.csv
DOWNSTREAM_DIR=/workspace/PRISM/results/c4_downstream/downstream
DONEDIR=$OUT/done
LOGDIR=logs
PROGRESS=$OUT/progress.tsv
mkdir -p "$OUT" "$DOWNSTREAM_DIR" "$DONEDIR" "$LOGDIR"
[ -f "$PROGRESS" ] || printf 'technique\tprecision\tsparsity\tppl\tseconds\tstatus\n' > "$PROGRESS"

DS_FLAGS=(--downstream --downstream-tasks "${DS_TASKS:-all}")
[ -n "${DS_LIMIT:-}" ]     && DS_FLAGS+=(--downstream-limit "$DS_LIMIT")
[ -n "${DS_GEN_LIMIT:-}" ] && DS_FLAGS+=(--downstream-gen-limit "$DS_GEN_LIMIT")
[ -z "${DS_NO_CODEEXEC:-}" ] && DS_FLAGS+=(--downstream-allow-codeexec)
echo "[slimgrid] downstream flags: ${DS_FLAGS[*]}"

_SHARD="${DS_SHARD:-0/1}"; SHARD_IDX="${_SHARD%%/*}"; SHARD_TOTAL="${_SHARD##*/}"
echo "[slimgrid] shard ${SHARD_IDX}/${SHARD_TOTAL}  tasks=${DS_TASKS:-all}"

CFG_N=0; TOTAL=0; DONE=0
run() {  # technique precision sparsity
    local tech=$1 prec=$2 sp=$3
    local idx=$CFG_N; CFG_N=$((CFG_N+1))
    [ $((idx % SHARD_TOTAL)) -ne "$SHARD_IDX" ] && return
    local tag="${tech}_${prec}bit_$(python3 -c "print(int($sp*100))")sp"
    TOTAL=$((TOTAL+1))
    if [ -f "$DONEDIR/$tag.done" ]; then echo "[skip] $tag"; DONE=$((DONE+1)); return; fi
    local log="$LOGDIR/c4_${tag}.log"
    local t0=$(date +%s)
    "$PY" benchmarks/benchmark_suite.py \
        --model qwen-0.5b --technique "$tech" --precision "$prec" \
        --sparsity "$sp" --dataset c4 \
        "${DS_FLAGS[@]}" \
        --downstream-csv-dir "$DOWNSTREAM_DIR" \
        --csv "$MAIN_CSV" > "$log" 2>&1
    local rc=$?; local t1=$(date +%s)
    local ppl; ppl=$(grep -oE "Perplexity: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+")
    local status="ok"; [ $rc -ne 0 ] && status="rc=$rc"
    if [ -n "$ppl" ] && [ $rc -eq 0 ]; then touch "$DONEDIR/$tag.done"; DONE=$((DONE+1));
    else [ -z "$ppl" ] && status="${status},noppl"; fi
    [ -z "$ppl" ] && ppl="NA"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tech" "$prec" "$sp" "$ppl" "$((t1-t0))" "$status" >> "$PROGRESS"
    echo "[slimgrid] ${tag} -> ppl=${ppl} (${status}, $((t1-t0))s)"
    if [ "$rc" -eq 126 ] || [ $((t1-t0)) -lt 0 ]; then
        echo "[slimgrid] ENV-GLITCH (rc=$rc, dur=$((t1-t0))s) — aborting shard for supervisor retry"; exit 3
    fi
}

echo "[slimgrid] START $(date)"
# 9 SLiM configs (same precision/sparsity grid as PRISM). Order mirrors prism block.
for sp in 0.05 0.25 0.50; do
    for prec in 3 4 5; do run slim "$prec" "$sp"; done
done
echo "[slimgrid] DONE $(date)  summary: ${DONE}/${TOTAL} configs complete"
