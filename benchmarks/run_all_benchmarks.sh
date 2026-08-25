#!/bin/bash
# Comprehensive Benchmark Runner - All Methods, Precisions, Models
# After SINQ group-wise quantization fix

cd /workspace/SINQ
source env/bin/activate

echo "=============================================="
echo "Complete Benchmark Suite (Post-Fix)"
echo "=============================================="
echo "Models: qwen-0.5b, qwen-1.5b, qwen-3b, phi-2"
echo "Methods: fp16, wanda, sparsegpt, sinq, sinq-sparse"
echo "Precisions: 3, 4, 5, 6, 8 (where applicable)"
echo "Sparsity: 35% for sparse methods"
echo "=============================================="
echo ""

MODELS="qwen-0.5b qwen-1.5b qwen-3b phi-2"
PRECISIONS="3 4 5 6 8"

# ==========================================
# PHASE 1: FP16 BASELINES (no quantization)
# ==========================================
echo "=========================================="
echo "PHASE 1: FP16 BASELINES"
echo "=========================================="

for model in $MODELS; do
    echo ""
    echo "[FP16] Running $model..."
    python benchmarks/benchmark_suite.py --model $model --technique fp16 --precision 0
done

# ==========================================
# PHASE 2: WANDA (pruning only, no quantization)
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 2: WANDA BENCHMARKS (35% sparsity)"
echo "=========================================="

for model in $MODELS; do
    echo ""
    echo "[WANDA] Running $model..."
    python benchmarks/benchmark_suite.py --model $model --technique wanda --precision 0
done

# ==========================================
# PHASE 3: SINQ (quantization only)
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 3: SINQ BENCHMARKS (quantization only)"
echo "=========================================="

for model in $MODELS; do
    for precision in $PRECISIONS; do
        echo ""
        echo "[SINQ] Running $model at ${precision}-bit..."
        python benchmarks/benchmark_suite.py --model $model --technique sinq --precision $precision
    done
done

# ==========================================
# PHASE 4: SINQ-SPARSE (pruning + quantization)
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 4: SINQ-SPARSE BENCHMARKS (35% sparsity + quantization)"
echo "=========================================="

for model in $MODELS; do
    for precision in $PRECISIONS; do
        echo ""
        echo "[SINQ-Sparse] Running $model at ${precision}-bit..."
        python benchmarks/benchmark_suite.py --model $model --technique sinq-sparse --precision $precision
    done
done

# ==========================================
# PHASE 5: SPARSEGPT (pruning + quantization)
# ==========================================
echo ""
echo "=========================================="
echo "PHASE 5: SPARSEGPT BENCHMARKS (35% sparsity + quantization)"
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
echo "=============================================="
echo "Total configurations:"
echo "  - FP16: 4 models"
echo "  - Wanda: 4 models"
echo "  - SINQ: 4 models x 5 precisions = 20"
echo "  - SINQ-Sparse: 4 models x 5 precisions = 20"
echo "  - SparseGPT: 4 models x 5 precisions = 20"
echo "  - TOTAL: 68 configurations"
echo "Results saved to: /workspace/SINQ/results/"
echo "=============================================="
