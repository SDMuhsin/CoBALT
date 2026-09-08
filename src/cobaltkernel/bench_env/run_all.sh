#!/usr/bin/env bash
# Reproduce the measured hardware ceilings quoted in docs/REPRODUCTION.md and docs/KERNELS.md.
#   bash src/cobaltkernel/bench_env/run_all.sh [MIG-uuid ...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH=/scratch/root/PTQResearch/cuda-13/bin:$PATH
export TMPDIR=/scratch/root/PTQResearch/tmp
OUT=/scratch/root/PTQResearch/tmp/ckbench; mkdir -p "$OUT"
ARCH="-arch=sm_120a"
for f in props bw coop gemv4bit mma_test; do
  echo "[build] $f"
  nvcc -O3 $ARCH --std=c++17 -o "$OUT/$f" "$HERE/$f.cu" || exit 1
done
SLICES=("${@:-MIG-1d47bdbe-9b64-59b9-bae8-ae32bd1dfbe0 MIG-bef7a31e-4317-582c-a97d-75e9e429441c}")
for U in ${SLICES[@]}; do
  echo "==================== $U ===================="
  export CUDA_VISIBLE_DEVICES=$U
  "$OUT/props"
  "$OUT/bw" 8 5
  for KB in 0 24 48 96; do "$OUT/coop" $KB 256 1000; done
  "$OUT/mma_test"
  "$OUT/gemv4bit" 32 20
done
