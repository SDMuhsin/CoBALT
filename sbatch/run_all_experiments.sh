#!/bin/bash
# ============================================================================
# SINQ Comprehensive Experiment Suite - SLURM Submission Script
# Cluster: fir
# ============================================================================
#
# This script submits benchmark configurations based on the arrays below.
# All execution is controlled by modifying the CONFIGURATION arrays.
#
# Techniques (and what arrays they use):
#   - fp16:      FP16 baseline                    [models, datasets]
#   - sinq:      Quantization only                [models, precisions, datasets]
#   - wanda:     Pruning only (no quantization)   [models, sparsities, datasets]
#   - sparsegpt: Sparse + Quantization            [models, precisions, sparsities, datasets]
#   - prism:     Sparse-Aware + Quantization      [models, precisions, sparsities, datasets]
#
# Usage:
#   ./sbatch/run_all_experiments.sh                     # uses config below
#   ./sbatch/run_all_experiments.sh --hf-token TOKEN    # with HuggingFace token
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
# CONFIGURATION - Modify these arrays to control what runs
# ============================================================================

# Output CSV filename (results saved to ./results/${OUTPUT_FILE})
OUTPUT_FILE="${OUTPUT_FILE:-wikitext_gemma.csv}"

# Models to run
models=("gemma-2b")

# Techniques to run: fp16, sinq, sparsegpt, wanda, prism
techniques=("prism" "fp16" "wanda" "sparsegpt" "sinq")

# Precisions (used by sinq, sparsegpt, wanda, prism; ignored by fp16)
precisions=("3" "4" "5")

# Sparsity levels (used by sparsegpt, wanda, prism; ignored by fp16, sinq)
sparsities=("0.05" "0.25" "0.50")

# Datasets to evaluate on
datasets=("wikitext2")

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Job counter
job_count=0

# Create directories
mkdir -p ./logs ./results

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

get_job_config() {
    # Returns: gpu_type mem time_limit
    # Tailored to model size and dataset complexity
    local model=$1
    local dataset=$2

    # Base times per model (for PPL datasets like wikitext2)
    # MMLU takes ~2x longer due to multiple choice evaluation
    local base_time_min=0
    local gpu_type=""
    local mem=""

    case $model in
        "qwen-0.5b")
            # ~0.5B params, very fast
            gpu_type="nvidia_h100_80gb_hbm3_2g.20gb:1"
            mem="16000M"
            base_time_min=20
            ;;
        "opt-1.3b")
            # ~1.3B params
            gpu_type="nvidia_h100_80gb_hbm3_2g.20gb:1"
            mem="20000M"
            base_time_min=35
            ;;
        "gemma-2b")
            # ~2B params
            gpu_type="nvidia_h100_80gb_hbm3_3g.40gb:1"
            mem="32000M"
            base_time_min=25
            ;;
        "llama-7b")
            # ~7B params, needs full GPU
            gpu_type="h100:1"
            mem="64000M"
            base_time_min=50
            ;;
        *)
            # Default fallback
            gpu_type="nvidia_h100_80gb_hbm3_3g.40gb:1"
            mem="32000M"
            base_time_min=60
            ;;
    esac

    # Adjust time based on dataset
    local time_min=$base_time_min
    case $dataset in
        "wikitext2"|"ptb")
            # Standard PPL evaluation
            time_min=$base_time_min
            ;;
        "c4")
            # C4 is larger, ~1.5x time
            time_min=$((base_time_min * 3 / 2))
            ;;
        "mmlu")
            # MMLU is slow (many samples), ~2.5x time
            time_min=$((base_time_min * 5 / 2))
            ;;
    esac

    # Add 20% buffer and round up to nearest 5 minutes
    time_min=$((time_min * 12 / 10))
    time_min=$(( ((time_min + 4) / 5) * 5 ))

    # Cap minimum at 15 min, maximum at 6 hours
    if [ $time_min -lt 15 ]; then
        time_min=15
    elif [ $time_min -gt 360 ]; then
        time_min=360
    fi

    # Convert to HH:MM:SS
    local hours=$((time_min / 60))
    local mins=$((time_min % 60))
    local time_limit=$(printf "%02d:%02d:00" $hours $mins)

    echo "$gpu_type $mem $time_limit"
}

submit_job() {
    local model=$1
    local technique=$2
    local precision=$3
    local sparsity=$4
    local dataset=$5

    # Get tailored configuration
    read gpu_type mem time_limit <<< $(get_job_config $model $dataset)

    # Build job name
    if [[ -n "$sparsity" ]]; then
        sparsity_pct=$(echo "$sparsity * 100" | bc | cut -d'.' -f1)
        job_name="${model}_${technique}_${precision}b_sp${sparsity_pct}_${dataset}"
    else
        job_name="${model}_${technique}_${precision}b_${dataset}"
    fi

    log_file="./logs/${job_name}"

    # Build python command
    if [[ -n "$sparsity" ]]; then
        python_cmd="python benchmarks/benchmark_suite.py --model $model --technique $technique --precision $precision --sparsity $sparsity --dataset $dataset --csv $OUTPUT_FILE --quiet"
    else
        python_cmd="python benchmarks/benchmark_suite.py --model $model --technique $technique --precision $precision --dataset $dataset --csv $OUTPUT_FILE --quiet"
    fi

    # Add HF token if provided
    if [[ -n "$HF_TOKEN" ]]; then
        python_cmd="$python_cmd --hf-token $HF_TOKEN"
    fi

    echo "  $job_name [GPU: ${gpu_type%%:*}, Mem: $mem, Time: $time_limit]"

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
            # HF uses file locking for concurrent download safety
            export HF_HOME=\$(pwd)/cache
            export HF_DATASETS_CACHE=\$(pwd)/cache/datasets
            export TRANSFORMERS_CACHE=\$(pwd)/cache/transformers
            mkdir -p \$HF_HOME \$HF_DATASETS_CACHE \$TRANSFORMERS_CACHE

            echo '========================================'
            echo 'Job: $job_name'
            echo 'Model: $model'
            echo 'Technique: $technique'
            echo 'Precision: ${precision}-bit'
            echo 'Sparsity: $sparsity'
            echo 'Dataset: $dataset'
            echo 'Cache: '\$HF_HOME
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

# Technique classification helpers
# Returns: "none", "precision", "sparsity", "both"
get_technique_type() {
    local technique=$1
    case $technique in
        "fp16")
            echo "none"        # No precision loop, no sparsity
            ;;
        "sinq")
            echo "precision"   # Precision loop only
            ;;
        "wanda")
            echo "sparsity"    # Sparsity loop only (pruning without quantization)
            ;;
        "sparsegpt"|"prism")
            echo "both"        # Both precision and sparsity loops
            ;;
        *)
            echo "both"        # Default to full loops
            ;;
    esac
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

echo "============================================================================"
echo "SINQ Comprehensive Experiment Suite"
echo "============================================================================"
echo "Cluster: fir"
echo "Output: ./results/${OUTPUT_FILE}"
if [[ -n "$HF_TOKEN" ]]; then
    echo "HF Token: provided (${#HF_TOKEN} chars)"
else
    echo "HF Token: not provided"
fi
echo "Models: ${models[*]}"
echo "Techniques: ${techniques[*]}"
echo "Precisions: ${precisions[*]}"
echo "Sparsities: ${sparsities[*]} (for sparse methods)"
echo "Datasets: ${datasets[*]}"
echo "============================================================================"
echo ""

# Iterate through all combinations based on technique type
for technique in "${techniques[@]}"; do
    technique_type=$(get_technique_type "$technique")

    echo "=============================================="
    echo "Submitting: ${technique^^} jobs (type: $technique_type)"
    echo "=============================================="

    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do

            case $technique_type in
                "none")
                    # FP16: no precision loop, no sparsity
                    submit_job "$model" "$technique" "16" "" "$dataset"
                    ;;
                "precision")
                    # SINQ: precision loop only
                    for precision in "${precisions[@]}"; do
                        submit_job "$model" "$technique" "$precision" "" "$dataset"
                    done
                    ;;
                "sparsity")
                    # Wanda: sparsity loop only (FP16 weights, no quantization)
                    for sparsity in "${sparsities[@]}"; do
                        submit_job "$model" "$technique" "16" "$sparsity" "$dataset"
                    done
                    ;;
                "both")
                    # SparseGPT/PRISM: both precision and sparsity loops
                    for precision in "${precisions[@]}"; do
                        for sparsity in "${sparsities[@]}"; do
                            submit_job "$model" "$technique" "$precision" "$sparsity" "$dataset"
                        done
                    done
                    ;;
            esac

        done
    done
    echo ""
done

# ============================================================================
# SUMMARY
# ============================================================================

echo "============================================================================"
echo "SUBMISSION COMPLETE"
echo "============================================================================"
echo "Total jobs submitted: $job_count"
echo ""

# Calculate breakdown dynamically
n_models=${#models[@]}
n_prec=${#precisions[@]}
n_sparse=${#sparsities[@]}
n_data=${#datasets[@]}

echo "Job breakdown by technique:"
for technique in "${techniques[@]}"; do
    technique_type=$(get_technique_type "$technique")
    case $technique_type in
        "none")
            count=$((n_models * n_data))
            echo "  - ${technique}: $n_models models x $n_data datasets = $count jobs"
            ;;
        "precision")
            count=$((n_models * n_prec * n_data))
            echo "  - ${technique}: $n_models models x $n_prec precisions x $n_data datasets = $count jobs"
            ;;
        "sparsity")
            count=$((n_models * n_sparse * n_data))
            echo "  - ${technique}: $n_models models x $n_sparse sparsities x $n_data datasets = $count jobs"
            ;;
        "both")
            count=$((n_models * n_prec * n_sparse * n_data))
            echo "  - ${technique}: $n_models models x $n_prec prec x $n_sparse sparsities x $n_data datasets = $count jobs"
            ;;
    esac
done

echo ""
echo "Resource allocation by model:"
echo "  - qwen-0.5b: 20GB MIG slice, 16GB mem"
echo "  - opt-1.3b:  20GB MIG slice, 20GB mem"
echo "  - gemma-2b:  40GB MIG slice, 32GB mem"
echo "  - llama-7b:  Full 80GB H100, 64GB mem"
echo ""
echo "Logs: ./logs/"
echo "Results: ./results/${OUTPUT_FILE}"
echo "============================================================================"
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ./logs/<job_name>-*.out"
echo "============================================================================"
