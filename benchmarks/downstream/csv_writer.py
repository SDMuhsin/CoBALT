"""Thread-safe per-task CSV append.

Mirrors ``benchmark_suite.save_result_to_csv``: ``fcntl.flock`` LOCK_EX on an
``'a+'`` handle, auto-write the header when the file has none, ``csv.DictWriter``
with ``extrasaction='ignore'``, then flush + fsync before releasing the lock.

Header order is fixed by the task: config columns, then run columns
(limit, n_samples, duration_seconds, error), then the task's metric columns.
Dict-valued metrics (subject_accuracies / by_level / by_type) are json.dumps'd.
"""

import os
import csv
import json
import fcntl
from pathlib import Path

from .registry import (
    TASK_REGISTRY,
    CONFIG_COLUMNS,
    RUN_COLUMNS,
    JSON_METRIC_COLUMNS,
)


def fieldnames_for(task_key):
    """Full ordered header for a task's CSV: config + run + metric columns."""
    metric_columns = TASK_REGISTRY[task_key]["metric_columns"]
    return CONFIG_COLUMNS + RUN_COLUMNS + list(metric_columns)


def _normalize_row(task_key, row):
    """Coerce a row dict to CSV-writable values (json.dumps dict metrics, blanks)."""
    fieldnames = fieldnames_for(task_key)
    out = {}
    for col in fieldnames:
        val = row.get(col, "")
        if val is None:
            val = ""
        elif col in JSON_METRIC_COLUMNS and isinstance(val, (dict, list)):
            val = json.dumps(val)
        out[col] = val
    return out


def save_task_row(task_key, row, csv_path):
    """Append one row to ``csv_path`` for ``task_key`` (thread/process safe).

    Args:
        task_key: registry key (e.g. "hellaswag", "arc_easy").
        row: dict of column -> value (missing columns written blank).
        csv_path: Path or str to the per-task CSV file.
    """
    csv_path = Path(csv_path)
    fieldnames = fieldnames_for(task_key)
    out_row = _normalize_row(task_key, row)

    # Ensure parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Thread-safe write with file locking. 'a+' allows reading (header check)
    # and appending.
    with open(csv_path, 'a+', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            # Check if file already has a header.
            f.seek(0)
            first_line = f.readline()
            has_header = first_line.strip().startswith('timestamp')

            # Seek to end for appending.
            f.seek(0, 2)

            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')

            if not has_header:
                writer.writeheader()

            writer.writerow(out_row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
