"""Task registry: maps each downstream task key to its adapter, CSV filename,
metric columns, and a couple of flags the suite uses for orchestration.

Note on ARC: both ``arc_easy`` and ``arc_challenge`` point at the SAME
``eval_arc.run`` adapter (which evaluates both configs in one call). The suite
calls ``eval_arc`` ONCE and splits its {"ARC-Easy":..., "ARC-Challenge":...}
result into the two per-task CSV rows. The ``arc`` umbrella key carries
``expands_to`` so the suite knows the relationship.
"""

from . import eval_hellaswag
from . import eval_arc
from . import eval_humaneval
from . import eval_lambada
from . import eval_mmlu
from . import eval_math
from . import eval_mrr
from . import eval_piqa
from . import eval_winogrande
from . import eval_boolq
from . import eval_openbookqa


# The 8 task keys, in canonical order.
ALL_TASKS = [
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "humaneval",
    "lambada",
    "mmlu",
    "math",
    "mrr",
    "piqa",
    "winogrande",
    "boolq",
    "openbookqa",
]


# Generation-bound tasks (slow, ~10s/example) that the suite caps with a
# separate ``gen_limit`` so multiple-choice tasks can run full-split while
# these are subsampled independently.
GENERATIVE_TASKS = {"math", "humaneval"}


TASK_REGISTRY = {
    "hellaswag": {
        "adapter": eval_hellaswag.run,
        "csv": "hellaswag.csv",
        "metric_columns": ["accuracy", "correct", "total"],
        "expands_to": None,
        "needs_codeexec": False,
    },
    # ARC: single adapter, two CSV targets. The suite recognizes that
    # arc_easy/arc_challenge share an adapter and runs it once.
    "arc": {
        "adapter": eval_arc.run,
        "csv": None,  # umbrella key — no CSV of its own
        "metric_columns": ["accuracy", "correct", "total"],
        "expands_to": ["arc_easy", "arc_challenge"],
        "needs_codeexec": False,
    },
    "arc_easy": {
        "adapter": eval_arc.run,
        "csv": "arc_easy.csv",
        "metric_columns": ["accuracy", "correct", "total"],
        # Which key of the eval_arc result dict feeds this CSV.
        "arc_result_key": "ARC-Easy",
        "expands_to": None,
        "needs_codeexec": False,
    },
    "arc_challenge": {
        "adapter": eval_arc.run,
        "csv": "arc_challenge.csv",
        "metric_columns": ["accuracy", "correct", "total"],
        "arc_result_key": "ARC-Challenge",
        "expands_to": None,
        "needs_codeexec": False,
    },
    "humaneval": {
        "adapter": eval_humaneval.run,
        "csv": "humaneval.csv",
        "metric_columns": ["pass_at_1", "passed", "total"],
        "expands_to": None,
        "needs_codeexec": True,
    },
    "lambada": {
        "adapter": eval_lambada.run,
        "csv": "lambada.csv",
        "metric_columns": ["accuracy", "correct", "total"],
        "expands_to": None,
        "needs_codeexec": False,
    },
    "mmlu": {
        "adapter": eval_mmlu.run,
        "csv": "mmlu.csv",
        "metric_columns": ["accuracy", "correct", "total", "subject_accuracies"],
        "expands_to": None,
        "needs_codeexec": False,
    },
    "math": {
        "adapter": eval_math.run,
        "csv": "math.csv",
        "metric_columns": ["accuracy", "correct", "total", "by_level", "by_type"],
        "expands_to": None,
        "needs_codeexec": False,
    },
    "mrr": {
        "adapter": eval_mrr.run,
        "csv": "mrr.csv",
        "metric_columns": ["mrr", "total"],
        "expands_to": None,
        "needs_codeexec": False,
    },
    "piqa": {
        "adapter": eval_piqa.run,
        "csv": "piqa.csv",
        "metric_columns": ["accuracy", "correct", "total"],
        "expands_to": None,
        "needs_codeexec": False,
    },
    "winogrande": {
        "adapter": eval_winogrande.run,
        "csv": "winogrande.csv",
        "metric_columns": ["accuracy", "correct", "total"],
        "expands_to": None,
        "needs_codeexec": False,
    },
    "boolq": {
        "adapter": eval_boolq.run,
        "csv": "boolq.csv",
        "metric_columns": ["accuracy", "correct", "total"],
        "expands_to": None,
        "needs_codeexec": False,
    },
    "openbookqa": {
        "adapter": eval_openbookqa.run,
        "csv": "openbookqa.csv",
        "metric_columns": ["accuracy", "correct", "total"],
        "expands_to": None,
        "needs_codeexec": False,
    },
}


# Config columns shared by every per-task CSV (must match benchmark_results.csv
# join keys: model + technique + precision + sparsity + dataset + timestamp).
CONFIG_COLUMNS = [
    "timestamp",
    "model",
    "model_name",
    "technique",
    "precision",
    "sparsity",
    "dataset",
]

# Run-level columns that follow the config columns in every per-task CSV.
RUN_COLUMNS = ["limit", "n_samples", "duration_seconds", "error"]

# Metric columns that hold dict values and must be json.dumps'd on write.
JSON_METRIC_COLUMNS = {"subject_accuracies", "by_level", "by_type"}
