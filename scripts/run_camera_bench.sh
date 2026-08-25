#!/usr/bin/env bash
# Camera-ready benchmark campaign launcher.
#
# Sources env.sh (venv + HF offline cache) then runs the parallel MIG dispatcher,
# which fans the (method x sparsity) grid across free MIG slices, one cell per
# slice, resumably. All args pass through to scripts/camera_dispatch.py.
#
# Typical use (survives disconnect; watch the log):
#     nohup bash scripts/run_camera_bench.sh > results/benchmark_camera/dispatch.log 2>&1 &
#     tail -f results/benchmark_camera/dispatch.log
#
# Preview the plan without launching anything:
#     bash scripts/run_camera_bench.sh --dry-run
#
# Run a subset (priority-first):
#     bash scripts/run_camera_bench.sh --methods cobalt,sparsegpt,wanda-awq --sparsities 0.7
set -u
cd "$(dirname "$0")/.."
# The dispatcher itself uses no GPU; each cell subprocess pins its own MIG slice.
export CUDA_VISIBLE_DEVICES=""
source env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""   # env.sh auto-picks a slice; unset it for the scheduler
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p results/benchmark_camera/logs
exec python scripts/camera_dispatch.py "$@"
