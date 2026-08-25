"""Downstream-task evaluation package for the PRISM quantization benchmark.

Runs downstream benchmarks (HellaSwag, ARC, HumanEval, LAMBADA, MMLU, MATH, MRR)
on an in-memory quantized model right after PPL eval, and writes one CSV per
benchmark keyed by the run config. Scoring/loglikelihood/generation math is
ported faithfully from the CRB golden-reference repo; only the I/O boundary is
adapted to the uniform ``run(model, tokenizer, device, ...)`` adapter contract.

Public API:
    run_downstream_suite(model, tokenizer, device, *, tasks="all", limit=None,
                         allow_codeexec=False, config=None, results_dir=None,
                         seqlen=2048, verbose=False) -> list[dict]
    TASK_REGISTRY  — task_key -> {adapter, csv, metric_columns, expands_to, needs_codeexec}
    ALL_TASKS      — canonical list of the 8 task keys

Designed to work as a top-level package named ``downstream`` (benchmark_suite
inserts the ``benchmarks/`` dir on sys.path), so all intra-package imports are
relative.
"""

from .registry import TASK_REGISTRY, ALL_TASKS
from .suite import run_downstream_suite

__all__ = ["run_downstream_suite", "TASK_REGISTRY", "ALL_TASKS"]
