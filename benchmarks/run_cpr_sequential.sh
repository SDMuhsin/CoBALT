#!/bin/bash
# Sequential CPR benchmark runner
# Runs all CPR-SINQ and CPR-Sparse experiments one at a time

source /workspace/SINQ/env/bin/activate
cd /workspace/SINQ

MODELS=("qwen-0.5b" "qwen-1.5b" "qwen-3b" "phi-2")
TECHNIQUES=("cpr-sinq" "cpr-sparse")
PRECISIONS=(3 4 5 6)

echo "=============================================="
echo "Sequential CPR Benchmark Runner"
echo "Models: ${MODELS[*]}"
echo "Techniques: ${TECHNIQUES[*]}"
echo "Precisions: ${PRECISIONS[*]}"
echo "Total configs: $((${#MODELS[@]} * ${#TECHNIQUES[@]} * ${#PRECISIONS[@]}))"
echo "=============================================="

count=0
total=$((${#MODELS[@]} * ${#TECHNIQUES[@]} * ${#PRECISIONS[@]}))

for model in "${MODELS[@]}"; do
    for technique in "${TECHNIQUES[@]}"; do
        for precision in "${PRECISIONS[@]}"; do
            count=$((count + 1))
            echo ""
            echo "=============================================="
            echo "[$count/$total] Running: $model | $technique | ${precision}-bit"
            echo "=============================================="

            python benchmarks/benchmark_suite.py \
                --model "$model" \
                --technique "$technique" \
                --precision "$precision"

            # Check exit status
            if [ $? -eq 0 ]; then
                echo "Completed: $model | $technique | ${precision}-bit"
            else
                echo "FAILED: $model | $technique | ${precision}-bit"
            fi
        done
    done
done

echo ""
echo "=============================================="
echo "All benchmarks completed!"
echo "=============================================="
