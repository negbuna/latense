#!/bin/bash

# Configuration
DATASET="wics/strategy-qa"
LIMIT=500
OUTPUT_BASE="./results_full_baselines"
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it" "Qwen/Qwen2.5-7B-Instruct")
MODES=("greedy_cot" "self_consistency")

mkdir -p "$OUTPUT_BASE"

echo "========================================================"
echo "STARTING ALL STRATEGY-QA BASELINES (N=$LIMIT)"
echo "Dataset: $DATASET"
echo "========================================================"

for MODEL in "${MODELS[@]}"; do
    # Get a clean name for the output folder
    MODEL_CLEAN=$(echo "$MODEL" | cut -d'/' -f2)
    for MODE in "${MODES[@]}"; do
        echo "--------------------------------------------------------"
        echo "Running Model: $MODEL | Mode: $MODE"
        echo "--------------------------------------------------------"
        
        # We use --max_eval_samples $LIMIT for src/main.py
        python3 src/main.py \
            --model_name_or_path "$MODEL" \
            --dataset "$DATASET" \
            --generation_mode "$MODE" \
            --output_dir "$OUTPUT_BASE/${MODEL_CLEAN}_${MODE}" \
            --max_eval_samples $LIMIT \
            --seed 42 \
            --sc_k 5
        
        echo "Completed $MODEL | $MODE"
        echo "--------------------------------------------------------"
    done
done

echo "ALL BASELINES COMPLETED. Results are in $OUTPUT_BASE"
