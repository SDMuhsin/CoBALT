#!/bin/bash
# ============================================================================
# SINQ Sparsity Ablation Study - SLURM Submission Script
# Cluster: fir
# ============================================================================
#
# Ablates sparsity from 5% to 95% in steps of 5% for:
#   - Model: qwen-0.5b
#   - Techniques: Wanda (no quant), SparseGPT (3,5-bit), PRISM (3,5-bit)
#   - Dataset: wikitext2
#
# Usage:
#   ./sbatch/run_sparsity_ablation.sh                     # uses config below
#   ./sbatch/run_sparsity_ablation.sh --hf-token TOKEN    # with HuggingFace token
#
# ============================================================================

# Note: Don't use 'set -e' - it breaks ((job_count++)) when count is 0

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

HF_TOKEN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--hf-token TOKEN]"
            exit 1
            ;;
    esac
done

# ============================================================================
# CONFIGURATION
# ============================================================================

# Fixed model for ablation
MODEL="qwen-0.5b"

# Sparsity levels: 5% to 95% in steps of 5%
sparsities=("0.55" "0.60" "0.65" "0.70" "0.75" "0.80" "0.85" "0.90" "0.95") # ( ("0.05" "0.10" "0.15" "0.20" "0.25" "0.30" "0.35" "0.40" "0.45" "0.50")  

# Precisions for quantized methods
precisions=("3" "5")

# Dataset
DATASET="wikitext2"

# Output CSV (distinct for this ablation)
CSV_FILE="sparsity_ablation_results.csv"

# Job counter
job_count=0

# Create directories
mkdir -p ./logs ./results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_time_limit() {
    # qwen-0.5b is fast, but higher sparsity = slightly faster
    # Base: 20 min, add buffer
    echo "00:30:00"
}

submit_job() {
    local technique=$1
    local precision=$2
    local sparsity=$3

    local gpu_type="nvidia_h100_80gb_hbm3_2g.20gb:1"
    local mem="16000M"
    local time_limit=$(get_time_limit)

    # Build job name
    sparsity_pct=$(echo "$sparsity * 100" | bc | cut -d'.' -f1)
    if [[ "$precision" == "0" ]]; then
        job_name="ablate_${MODEL}_${technique}_sp${sparsity_pct}"
    else
        job_name="ablate_${MODEL}_${technique}_${precision}b_sp${sparsity_pct}"
    fi

    log_file="./logs/${job_name}"

    # Build python command
    if [[ "$precision" == "0" ]]; then
        # Wanda: pruning only, no quantization
        python_cmd="python benchmarks/benchmark_suite.py --model $MODEL --technique $technique --precision 16 --sparsity $sparsity --dataset $DATASET --csv $CSV_FILE --quiet"
    else
        python_cmd="python benchmarks/benchmark_suite.py --model $MODEL --technique $technique --precision $precision --sparsity $sparsity --dataset $DATASET --csv $CSV_FILE --quiet"
    fi

    # Add HF token if provided
    if [[ -n "$HF_TOKEN" ]]; then
        python_cmd="$python_cmd --hf-token $HF_TOKEN"
    fi

    echo "  $job_name [Time: $time_limit]"

    sbatch \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --gpus=$gpu_type \
        --mem=$mem \
        --time=$time_limit \
        --output=${log_file}-%N-%j.out \
        --error=${log_file}-%N-%j.err \
        --job-name=$job_name \
        --wrap="
            module load scipy-stack cuda cudnn
            module load arrow
            source ./env/bin/activate

            # Shared cache for HuggingFace (datasets & models)
            export HF_HOME=\$(pwd)/cache
            export HF_DATASETS_CACHE=\$(pwd)/cache/datasets
            export TRANSFORMERS_CACHE=\$(pwd)/cache/transformers
            mkdir -p \$HF_HOME \$HF_DATASETS_CACHE \$TRANSFORMERS_CACHE

            echo '========================================'
            echo 'Sparsity Ablation Study'
            echo 'Job: $job_name'
            echo 'Model: $MODEL'
            echo 'Technique: $technique'
            echo 'Precision: $precision'
            echo 'Sparsity: $sparsity'
            echo 'Dataset: $DATASET'
            echo 'CSV: $CSV_FILE'
            echo 'Started: '\$(date)
            echo '========================================'
            nvidia-smi
            export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
            $python_cmd
            echo '========================================'
            echo 'Finished: '\$(date)
            echo '========================================'
        "

    ((job_count++))
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "============================================================================"
echo "SINQ Sparsity Ablation Study"
echo "============================================================================"
echo "Model: $MODEL"
echo "Techniques: Wanda (no quant), SparseGPT (3,5-bit), PRISM (3,5-bit)"
echo "Sparsities: 5% to 95% (step 5%)"
echo "Dataset: $DATASET"
echo "Output CSV: $CSV_FILE"
if [[ -n "$HF_TOKEN" ]]; then
    echo "HF Token: provided (${#HF_TOKEN} chars)"
else
    echo "HF Token: not provided"
fi
echo "============================================================================"
echo ""

# ============================================================================
# SECTION 1: WANDA (Pruning only - no quantization)
# ============================================================================
echo "=============================================="
echo "Section 1: Wanda (Pruning Only)"
echo "=============================================="

for sparsity in "${sparsities[@]}"; do
    submit_job "wanda" "0" "$sparsity"
    echo "Wanda Loop"
done

# ============================================================================
# SECTION 2: SPARSEGPT (Pruning + Quantization)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 2: SparseGPT (3-bit and 5-bit)"
echo "=============================================="

for precision in "${precisions[@]}"; do
    for sparsity in "${sparsities[@]}"; do
        submit_job "sparsegpt" "$precision" "$sparsity"
	echo "SparseGPT Loop"
    done
done

# ============================================================================
# SECTION 3: PRISM (Pruning + Quantization)
# ============================================================================
echo ""
echo "=============================================="
echo "Section 3: PRISM (3-bit and 5-bit)"
echo "=============================================="

for precision in "${precisions[@]}"; do
    for sparsity in "${sparsities[@]}"; do
        submit_job "prism" "$precision" "$sparsity"
    done
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "============================================================================"
echo "SUBMISSION COMPLETE"
echo "============================================================================"
echo "Total jobs submitted: $job_count"
echo ""
echo "Job breakdown:"
n_sparse=${#sparsities[@]}
n_prec=${#precisions[@]}
echo "  - Wanda:     $n_sparse sparsities x 1 = $n_sparse jobs"
echo "  - SparseGPT: $n_sparse sparsities x $n_prec precisions = $((n_sparse * n_prec)) jobs"
echo "  - PRISM:     $n_sparse sparsities x $n_prec precisions = $((n_sparse * n_prec)) jobs"
echo "  - TOTAL:     $((n_sparse + n_sparse * n_prec * 2)) jobs"
echo ""
echo "Results will be saved to: ./results/$CSV_FILE"
echo "============================================================================"
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ./logs/ablate_*.out"
echo "============================================================================"
