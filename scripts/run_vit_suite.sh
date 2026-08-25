#!/usr/bin/env bash
# Full bit-matched TUNED ViT/ImageNet suite (2 ViT-large models, top-1 on a 10k subset).
# Same 27-hp-cell/(bits,sp) tuned+matched grid as the GLUE suites. Both models + ImageNet-val
# parquet are already cached, so this runs fully offline.
#
# Launch (survives disconnect):
#     mkdir -p results/benchmark_camera_vit_tuned
#     nohup bash scripts/run_vit_suite.sh \
#         > results/benchmark_camera_vit_tuned/dispatch.log 2>&1 &
# Preview:
#     bash scripts/run_vit_suite.sh --dry-run
set -u
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=""
source env.sh >/dev/null 2>&1
export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec python scripts/dispatch_vit_suite.py "$@"
