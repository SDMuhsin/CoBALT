#!/bin/bash
# LLaMA-7B downstream SANITY sweep for the recently-added baselines (validated only on
# qwen-0.5b before). Goal: confirm the downstream results on LLaMA are NOT anomalous
# (sane vs the fp16 anchor + trusted references), NOT paper-grade full-split numbers.
#
# Representative operating point: 4-bit. Quant-only methods at 0% sparsity; sparse+quant
# methods at 50%. Trusted references (fp16/sinq/wanda/sparsegpt/prism) included as anchors.
# Downstream subsampled (DS_LIMIT, default 256) for tractability on 7B.
#
# Same per-config-subprocess + done-marker + glitch-guard design as run_c4_downstream_grid.sh.
# Outputs under results/llama_downstream/. ENV: PRISM_MIG, DS_TASKS, DS_LIMIT, DS_GEN_LIMIT,
# DS_NO_CODEEXEC, DS_SHARD="idx/total", DATASET (default wikitext2).
set -u
cd /workspace/PRISM

export HF_HOME=/workspace/PRISM/cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=1
unset PYTHONPATH
export PIP_CONFIG_FILE=/dev/null

if [ -n "${PRISM_MIG:-}" ]; then
    MIG="$PRISM_MIG"; echo "[llamagrid] PRISM_MIG=$MIG"
else
    MIG="$(python scripts/pick_free_mig.py)"; [ -n "$MIG" ] && echo "[llamagrid] auto MIG=$MIG"
fi
[ -n "$MIG" ] && export CUDA_VISIBLE_DEVICES="$MIG"
echo "[llamagrid] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

PY=/workspace/PRISM/env/bin/python
DATASET="${DATASET:-wikitext2}"
OUT=results/llama_downstream
MAIN_CSV=llama_downstream/main.csv
DOWNSTREAM_DIR=/workspace/PRISM/results/llama_downstream/downstream
DONEDIR=$OUT/done
LOGDIR=logs
PROGRESS=$OUT/progress.tsv
mkdir -p "$OUT" "$DOWNSTREAM_DIR" "$DONEDIR" "$LOGDIR"
[ -f "$PROGRESS" ] || printf 'technique\tprecision\tsparsity\tppl\tseconds\tstatus\n' > "$PROGRESS"

DS_FLAGS=(--downstream --downstream-tasks "${DS_TASKS:-all}" --downstream-limit "${DS_LIMIT:-256}")
[ -n "${DS_GEN_LIMIT:-}" ] && DS_FLAGS+=(--downstream-gen-limit "$DS_GEN_LIMIT")
[ -z "${DS_NO_CODEEXEC:-}" ] && DS_FLAGS+=(--downstream-allow-codeexec)
echo "[llamagrid] downstream flags: ${DS_FLAGS[*]}  dataset=$DATASET"

_SHARD="${DS_SHARD:-0/1}"; SHARD_IDX="${_SHARD%%/*}"; SHARD_TOTAL="${_SHARD##*/}"
echo "[llamagrid] shard ${SHARD_IDX}/${SHARD_TOTAL}"

CFG_N=0; TOTAL=0; DONE=0
run() {  # technique precision sparsity
    local tech=$1 prec=$2 sp=$3
    local idx=$CFG_N; CFG_N=$((CFG_N+1))
    [ $((idx % SHARD_TOTAL)) -ne "$SHARD_IDX" ] && return
    local tag="${tech}_${prec}bit_$(python3 -c "print(int($sp*100))")sp"
    TOTAL=$((TOTAL+1))
    if [ -f "$DONEDIR/$tag.done" ]; then echo "[skip] $tag"; DONE=$((DONE+1)); return; fi
    local log="$LOGDIR/llama_${tag}.log"
    local t0=$(date +%s)
    "$PY" benchmarks/benchmark_suite.py \
        --model llama-7b --technique "$tech" --precision "$prec" \
        --sparsity "$sp" --dataset "$DATASET" \
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
    echo "[llamagrid] ${tag} -> ppl=${ppl} (${status}, $((t1-t0))s)"
    if [ "$rc" -eq 126 ] || [ $((t1-t0)) -lt 0 ]; then
        echo "[llamagrid] ENV-GLITCH (rc=$rc, dur=$((t1-t0))s) — aborting shard"; exit 3
    fi
}

echo "[llamagrid] START $(date)  dataset=$DATASET limit=${DS_LIMIT:-256}"
# Anchor + quant-only group (0% sparsity, 4-bit).
run fp16 4 0.0
run sinq 4 0.0
run awq  4 0.0
run gptq 4 0.0
run spqr 4 0.0
# Sparse + (sparse+quant) group (50% sparsity, 4-bit).
run wanda     4 0.5
run sparsegpt 4 0.5
run prism     4 0.5
run wanda-awq  4 0.5
run wanda-sinq 4 0.5
run jsq    4 0.5
run jsq-wo 4 0.5
run slim   4 0.5
echo "[llamagrid] DONE $(date)  summary: ${DONE}/${TOTAL} configs"
