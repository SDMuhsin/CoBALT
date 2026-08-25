#!/bin/bash
# Reproduce downstream MMLU (5-shot) for a prune+quant technique on gemma-2b
# at 3-bit / 75% sparsity.
#
#   ./scripts/run_mmlu_sp75.sh <technique> [downstream_limit]
#
# technique: prism | wanda-sinq | wanda-awq (any TECHNIQUES entry)
# downstream_limit: optional; subsample N MMLU test questions (smoke tests).
#
# Writes per-task CSVs to results/downstream_sp75/ and the run JSON to results/.
set -euo pipefail

REPO=/workspace/PTQResearch
# shellcheck disable=SC1091
source "$REPO/env.sh"
cd "$REPO"

TECH="${1:?usage: scripts/run_mmlu_sp75.sh <technique> [limit]}"
LIMIT_ARG=()
[ "${2:-}" != "" ] && LIMIT_ARG=(--downstream-limit "$2")

python benchmarks/benchmark_suite.py \
    --model gemma-2b \
    --technique "$TECH" \
    --precision 3 \
    --sparsity 0.75 \
    --dataset wikitext2 \
    --downstream \
    --downstream-tasks mmlu \
    --downstream-csv-dir "$REPO/results/downstream_sp75" \
    "${LIMIT_ARG[@]}"
