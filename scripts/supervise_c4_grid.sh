#!/bin/bash
# Supervisor for the C4 + downstream grid. Keeps relaunching the two shards
# until all 25 configs have done-markers, riding out the box's periodic glitches
# (clock jumps / rc=126 "cannot execute" that kill in-flight configs).
#
# Resume-safe: each round the shards skip configs that already have a
# done/<tag>.done marker, so completed work is never redone.
#
# Launch DETACHED so it survives a Claude session teardown:
#   cd /workspace/PRISM
#   setsid nohup bash scripts/supervise_c4_grid.sh >> logs/supervisor.log 2>&1 </dev/null &
#
# A box-wide GPU/driver crash will still kill this supervisor; just relaunch it.
set -u
cd /workspace/PRISM

DONEDIR=results/c4_downstream/done
PY=/workspace/PRISM/env/bin/python
TASKS=hellaswag,arc_easy,arc_challenge,humaneval,lambada,mmlu,mrr
# Two isolated MIG slices on GPU 0 (other users have been on GPU 1).
SLICE0=MIG-bef7a31e-4317-582c-a97d-75e9e429441c
SLICE1=MIG-6ec7b494-8fdc-5531-8226-d8b3ea71838a
TARGET=25
MAXROUNDS=60

ndone() { ls -1 "$DONEDIR" 2>/dev/null | wc -l | tr -d ' '; }

echo "[supervisor] START $(date)  done=$(ndone)/$TARGET"

for r in $(seq 1 "$MAXROUNDS"); do
    d=$(ndone)
    if [ "$d" -ge "$TARGET" ]; then
        echo "[supervisor] ALL $TARGET CONFIGS DONE at round $r $(date)"
        break
    fi
    echo "[supervisor] round $r begin $(date)  done=$d/$TARGET"

    # Wait until the GPU + torch are actually usable (box may still be recovering).
    if ! nvidia-smi -L >/dev/null 2>&1 || \
       ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        echo "[supervisor] GPU/torch not ready — waiting 120s"
        sleep 120
        continue
    fi

    PRISM_MIG="$SLICE0" DS_SHARD=0/2 DS_TASKS="$TASKS" \
        bash scripts/run_c4_downstream_grid.sh >> logs/grid_shard0.log 2>&1 &
    P0=$!
    PRISM_MIG="$SLICE1" DS_SHARD=1/2 DS_TASKS="$TASKS" \
        bash scripts/run_c4_downstream_grid.sh >> logs/grid_shard1.log 2>&1 &
    P1=$!
    wait "$P0" "$P1"

    after=$(ndone)
    echo "[supervisor] round $r end $(date)  done=$after/$TARGET (+$((after-d)))"

    # If a round made zero progress (glitch/outage), back off before retrying.
    if [ "$after" -le "$d" ]; then
        echo "[supervisor] no progress this round — backing off 90s"
        sleep 90
    else
        sleep 15
    fi
done

echo "[supervisor] EXIT $(date)  done=$(ndone)/$TARGET"
