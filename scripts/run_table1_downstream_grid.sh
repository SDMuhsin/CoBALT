#!/usr/bin/env bash
# Supervisor for run_table1_downstream_grid.py — crash-proof way to run the full
# downstream sweep. The Python scheduler is resumable (skips configs whose task
# rows are present) and retries transient failures; this wrapper re-launches it
# until it exits 0, so you can start it once and walk away.
#
# Usage:
#   bash scripts/run_table1_downstream_grid.sh                 # full split, unattended
#   bash scripts/run_table1_downstream_grid.sh --ds-limit 200  # any args pass through
#   setsid bash scripts/run_table1_downstream_grid.sh >> logs/table1_ds_supervisor.log 2>&1 &
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/env/bin/python"
SCRIPT="${ROOT}/scripts/run_table1_downstream_grid.py"
MAX_LOOPS="${MAX_LOOPS:-1000}"

cd "${ROOT}" || exit 1

i=0
while [ "${i}" -lt "${MAX_LOOPS}" ]; do
    i=$((i + 1))
    echo "[supervisor] launch #${i} $(date -u +%H:%M:%S) : ${PY} ${SCRIPT} $*"
    "${PY}" "${SCRIPT}" "$@"
    rc=$?
    if [ "${rc}" -eq 0 ]; then
        echo "[supervisor] scheduler exited 0 — downstream grid complete."
        exit 0
    fi
    echo "[supervisor] scheduler exited ${rc}; some configs unfinished. retry in 15s..."
    sleep 15
done

echo "[supervisor] hit MAX_LOOPS=${MAX_LOOPS}; giving up. re-run to continue."
exit 1
