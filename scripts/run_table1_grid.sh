#!/usr/bin/env bash
# Supervisor for run_table1_grid.py — the crash-proof way to run the full grid.
#
# The Python scheduler is already resumable (it skips any config whose result row
# is present) and retries transient failures. This wrapper adds the outer loop for
# when the whole process dies — a box glitch, an OOM kill, a session restart. It
# just re-launches the scheduler until it exits 0 (all requested cells done), so
# you can start it once and walk away.
#
# Usage:
#   bash scripts/run_table1_grid.sh                 # run to completion, unattended
#   bash scripts/run_table1_grid.sh --only gemma    # any run_table1_grid.py args pass through
#
# Detach it so it survives your shell/session dying:
#   setsid bash scripts/run_table1_grid.sh >> logs/table1_grid_supervisor.log 2>&1 &
#
# Re-running it later is safe and cheap: finished cells are skipped instantly.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/env/bin/python"
SCRIPT="${ROOT}/scripts/run_table1_grid.py"
MAX_LOOPS="${MAX_LOOPS:-1000}"   # backstop against an infinite crash loop

cd "${ROOT}" || exit 1

i=0
while [ "${i}" -lt "${MAX_LOOPS}" ]; do
    i=$((i + 1))
    echo "[supervisor] launch #${i} $(date -u +%H:%M:%S) : ${PY} ${SCRIPT} $*"
    "${PY}" "${SCRIPT}" "$@"
    rc=$?
    if [ "${rc}" -eq 0 ]; then
        echo "[supervisor] scheduler exited 0 — grid complete."
        exit 0
    fi
    echo "[supervisor] scheduler exited ${rc}; some cells unfinished. retrying in 15s..."
    sleep 15
done

echo "[supervisor] hit MAX_LOOPS=${MAX_LOOPS}; giving up. re-run to continue."
exit 1
