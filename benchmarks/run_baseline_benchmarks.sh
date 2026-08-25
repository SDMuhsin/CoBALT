#!/bin/bash
# Comprehensive Wanda and SparseGPT Benchmark Runner
# Runs all models across all applicable precisions

cd /workspace/SINQ
source env/bin/activate

echo "=============================================="
echo "Wanda & SparseGPT Comprehensive Benchmark Suite"
echo "=============================================="
echo "Models: qwen-0.5b, qwen-1.5b, qwen-3b, phi-2"
echo "Wanda: 4 models (pruning only)"
echo "SparseGPT: 4 models x 5 precisions = 20 configs"
echo "Total: 24 configurations"
echo "=============================================="
echo ""

# Skip llama-7b as it requires more memory and is often gated
MODELS="qwen-0.5b qwen-1.5b qwen-3b phi-2"
PRECISIONS="3 4 5 6 8"

# ==========================================
# WANDA BENCHMARKS (pruning only, no precision)
# ==========================================
echo "=========================================="
echo "PHASE 1: WANDA BENCHMARKS"
echo "=========================================="

for model in $MODELS; do
    echo ""
    echo "[WANDA] Running $model..."
    python benchmarks/benchmark_suite.py --model $model --technique wanda --precision 0
done

# ==========================================
# SPARSEGPT BENCHMARKS (pruning + quantization)
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 2: SPARSEGPT BENCHMARKS"
echo "=========================================="

for model in $MODELS; do
    for precision in $PRECISIONS; do
        echo ""
        echo "[SparseGPT] Running $model at ${precision}-bit..."
        python benchmarks/benchmark_suite.py --model $model --technique sparsegpt --precision $precision
    done
done

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "Results saved to: /workspace/SINQ/results/"
echo "=============================================="
