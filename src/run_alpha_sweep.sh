#!/bin/bash

# Usage: ./src/run_alpha_sweep.sh <dataset> <model> <cluster_dir> <gpu_id> <split>
# Example: ./src/run_alpha_sweep.sh "openai/gsm8k" "meta-llama/Llama-3.1-8B-Instruct" "./vectors/gsm8k_llama" 0 "train[:200]"

DATASET=$1
MODEL=$2
CLUSTER_DIR=$3
GPU_ID=$4
SPLIT=${5:-dev}

# Define alphas to test (Standard range for dynamic scaling)
ALPHAS=(0.1 0.2 0.3 0.4 0.5)
BEST_ALPHA=0
BEST_ACC=0
RESULTS_LOG=""

echo "========================================"
echo "Starting Alpha Sweep"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Using Split: $SPLIT (200 samples) with Dynamic Scaling"
echo "========================================"

for alpha in "${ALPHAS[@]}"; do
    echo -n "Testing alpha=$alpha... "
    
    # Run eval on a small subset of dev
    # We use --use_dynamic_scaling by default as per your preference
    OUTPUT=$(CUDA_VISIBLE_DEVICES=$GPU_ID python src/main.py \
        --model_name_or_path "$MODEL" \
        --dataset "$DATASET" \
        --generation_mode "latense" \
        --cluster_dir "$CLUSTER_DIR" \
        --output_dir "./results/sweep/$(basename $CLUSTER_DIR)" \
        --split "$SPLIT" \
        --max_eval_samples 200 \
        --alpha $alpha \
        --verbose False 2>&1)
        
    # Extract accuracy
    # Output format: "Prompt 1 Final accuracy: 0.1234"
    ACC=$(echo "$OUTPUT" | grep "Final accuracy:" | tail -n 1 | sed -E 's/.*Final accuracy: ([0-9]+(\.[0-9]+)?).*/\1/')
    
    if [ -z "$ACC" ]; then
        echo "Failed (No accuracy found in output)"
        continue
    fi
    
    echo "Accuracy: $ACC"
    RESULTS_LOG+="\nAlpha $alpha: $ACC"
    
    # Compare using python since bash doesn't handle floats well
    IS_BETTER=$(python -c "print(1 if float($ACC) > float($BEST_ACC) else 0)")
    
    if [ "$IS_BETTER" -eq 1 ]; then
        BEST_ACC=$ACC
        BEST_ALPHA=$alpha
    fi
done

echo "========================================"
echo "Sweep Complete."
echo -e "Results Summary:$RESULTS_LOG"
echo "----------------------------------------"
echo "Best Alpha: $BEST_ALPHA (Accuracy: $BEST_ACC)"
echo "Recommendation: Use --alpha $BEST_ALPHA --use_dynamic_scaling for your main run."
echo "========================================"