#!/bin/bash
# Wanda Benchmark Runner - Run Wanda pruning on target models

cd /workspace/SINQ
source env/bin/activate

echo "=============================================="
echo "Wanda Benchmark Runner"
echo "Target: Qwen2.5-1.5B at 35% sparsity"
echo "=============================================="

# Run Wanda on qwen-0.5b first (validation)
echo ""
echo "[1/3] Validating Wanda on qwen-0.5b..."
python benchmarks/benchmark_suite.py --model qwen-0.5b --technique wanda --precision 0

# Run Wanda on qwen-1.5b (main comparison target)
echo ""
echo "[2/3] Running Wanda on qwen-1.5b (main target)..."
python benchmarks/benchmark_suite.py --model qwen-1.5b --technique wanda --precision 0

# Run Wanda on qwen-3b (extra comparison)
echo ""
echo "[3/3] Running Wanda on qwen-3b..."
python benchmarks/benchmark_suite.py --model qwen-3b --technique wanda --precision 0

echo ""
echo "=============================================="
echo "Wanda Benchmark Complete!"
echo "Results saved to: /workspace/SINQ/results/"
echo "=============================================="
