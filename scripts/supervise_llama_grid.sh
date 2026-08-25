#!/bin/bash
# Supervisor for the LLaMA-7B downstream sanity sweep (13 configs). Relaunches 4
# sharded run_llama_downstream_grid.sh instances until all 13 done-markers exist,
# riding out box glitches (clock jumps / rc=126). Resume-safe via done-markers.
#
# Launch DETACHED:
#   cd /workspace/PRISM
#   setsid nohup bash scripts/supervise_llama_grid.sh >> logs/llama_supervisor.log 2>&1 </dev/null &
set -u
cd /workspace/PRISM

DONEDIR=results/llama_downstream/done
PY=/workspace/PRISM/env/bin/python
TASKS="${DS_TASKS:-hellaswag,arc_easy,arc_challenge,humaneval,lambada,mmlu,mrr}"
LIMIT="${DS_LIMIT:-256}"
# 4 isolated 24GB MIG slices on GPU 0 (fit llama-7b fp16 = 13GB).
SLICES=(MIG-bef7a31e-4317-582c-a97d-75e9e429441c \
        MIG-6ec7b494-8fdc-5531-8226-d8b3ea71838a \
        MIG-475dbed1-782f-5c45-ae0c-1d2507268638 \
        MIG-12daede5-c316-57d7-bd83-a55c417864cd)
NSHARD=${#SLICES[@]}
TARGET=13
MAXROUNDS=40

ndone() { ls -1 "$DONEDIR" 2>/dev/null | wc -l | tr -d ' '; }
echo "[llama-sup] START $(date)  done=$(ndone)/$TARGET  nshard=$NSHARD limit=$LIMIT"
for r in $(seq 1 "$MAXROUNDS"); do
    d=$(ndone)
    if [ "$d" -ge "$TARGET" ]; then echo "[llama-sup] ALL $TARGET DONE at round $r $(date)"; break; fi
    echo "[llama-sup] round $r begin $(date)  done=$d/$TARGET"
    if ! nvidia-smi -L >/dev/null 2>&1 || \
       ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        echo "[llama-sup] GPU/torch not ready — waiting 120s"; sleep 120; continue
    fi
    pids=()
    for i in $(seq 0 $((NSHARD-1))); do
        PRISM_MIG="${SLICES[$i]}" DS_SHARD="$i/$NSHARD" DS_TASKS="$TASKS" DS_LIMIT="$LIMIT" \
            DS_GEN_LIMIT="${DS_GEN_LIMIT:-64}" \
            bash scripts/run_llama_downstream_grid.sh >> "logs/llama_shard$i.log" 2>&1 &
        pids+=($!)
    done
    wait "${pids[@]}"
    after=$(ndone)
    echo "[llama-sup] round $r end $(date)  done=$after/$TARGET (+$((after-d)))"
    if [ "$after" -le "$d" ]; then echo "[llama-sup] no progress — backing off 90s"; sleep 90; else sleep 15; fi
done
echo "[llama-sup] EXIT $(date)  done=$(ndone)/$TARGET"
