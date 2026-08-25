#!/bin/bash
# Supervisor for the SLiM-LoRA C4 + downstream grid (9 configs). Relaunches 3
# sharded run_slim_downstream_grid.sh instances until all 9 slim done-markers
# exist, riding out the box's periodic glitches (clock jumps / rc=126).
#
# Launch DETACHED so it survives a Claude session teardown:
#   cd /workspace/PRISM
#   setsid nohup bash scripts/supervise_slim_grid.sh >> logs/slim_supervisor.log 2>&1 </dev/null &
set -u
cd /workspace/PRISM

DONEDIR=results/c4_downstream/done
PY=/workspace/PRISM/env/bin/python
TASKS=hellaswag,arc_easy,arc_challenge,humaneval,lambada,mmlu,mrr
# 3 isolated MIG slices on GPU 0 (all free as of 2026-06-23).
SLICE0=MIG-bef7a31e-4317-582c-a97d-75e9e429441c
SLICE1=MIG-6ec7b494-8fdc-5531-8226-d8b3ea71838a
SLICE2=MIG-475dbed1-782f-5c45-ae0c-1d2507268638
TARGET=9
MAXROUNDS=60

ndone() { ls -1 "$DONEDIR" 2>/dev/null | grep -c '^slim_' | tr -d ' '; }

echo "[slim-sup] START $(date)  slim_done=$(ndone)/$TARGET"
for r in $(seq 1 "$MAXROUNDS"); do
    d=$(ndone)
    if [ "$d" -ge "$TARGET" ]; then
        echo "[slim-sup] ALL $TARGET SLiM CONFIGS DONE at round $r $(date)"; break
    fi
    echo "[slim-sup] round $r begin $(date)  slim_done=$d/$TARGET"
    if ! nvidia-smi -L >/dev/null 2>&1 || \
       ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        echo "[slim-sup] GPU/torch not ready — waiting 120s"; sleep 120; continue
    fi
    PRISM_MIG="$SLICE0" DS_SHARD=0/3 DS_TASKS="$TASKS" \
        bash scripts/run_slim_downstream_grid.sh >> logs/slim_shard0.log 2>&1 & P0=$!
    PRISM_MIG="$SLICE1" DS_SHARD=1/3 DS_TASKS="$TASKS" \
        bash scripts/run_slim_downstream_grid.sh >> logs/slim_shard1.log 2>&1 & P1=$!
    PRISM_MIG="$SLICE2" DS_SHARD=2/3 DS_TASKS="$TASKS" \
        bash scripts/run_slim_downstream_grid.sh >> logs/slim_shard2.log 2>&1 & P2=$!
    wait "$P0" "$P1" "$P2"
    after=$(ndone)
    echo "[slim-sup] round $r end $(date)  slim_done=$after/$TARGET (+$((after-d)))"
    if [ "$after" -le "$d" ]; then echo "[slim-sup] no progress — backing off 90s"; sleep 90; else sleep 15; fi
done
echo "[slim-sup] EXIT $(date)  slim_done=$(ndone)/$TARGET"
