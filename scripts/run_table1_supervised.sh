#!/bin/bash
# Supervisor for the Table-1 baseline grid scheduler.
#
# The scheduler (run_table1_baselines.py) already load-balances across MIG slices
# and retries each config RETRIES times. This wrapper adds an OUTER loop: it keeps
# re-invoking the scheduler (which resumes instantly via done-markers) until it
# exits 0 (no failed configs) or MAXROUNDS is hit — riding out the shared box's
# periodic glitches (clock jumps, a slice grabbed by another job, transient OOM).
#
# Launch DETACHED so it survives a session teardown:
#   cd /workspace/PRISM
#   setsid nohup bash scripts/run_table1_supervised.sh --exclude gemma \
#       >> logs/table1_supervisor_phase1.log 2>&1 </dev/null &
#
# All args are passed straight through to the scheduler (--exclude / --only / ...).
set -u
cd /workspace/PRISM
unset PYTHONPATH
export PIP_CONFIG_FILE=/dev/null
PY=/workspace/PRISM/env/bin/python
MAXROUNDS="${MAXROUNDS:-12}"

echo "[t1-sup] START $(date)  args: $*"
for r in $(seq 1 "$MAXROUNDS"); do
    echo "[t1-sup] ===== round $r/$MAXROUNDS $(date) ====="
    if ! nvidia-smi -L >/dev/null 2>&1 || \
       ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        echo "[t1-sup] GPU/torch not ready — waiting 120s"; sleep 120; continue
    fi
    "$PY" scripts/run_table1_baselines.py "$@"
    rc=$?
    echo "[t1-sup] round $r scheduler exit rc=$rc $(date)"
    if [ "$rc" -eq 0 ]; then
        echo "[t1-sup] ALL CONFIGS COMPLETE at round $r $(date)"; break
    fi
    echo "[t1-sup] some configs still failing — backing off 30s then resuming"; sleep 30
done
echo "[t1-sup] EXIT $(date)"
