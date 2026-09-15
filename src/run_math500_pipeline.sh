#!/bin/bash

# Configuration for MATH-500
LIMIT=200
SWEEP_SAMPLES=50
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it" "Qwen/Qwen2.5-7B-Instruct")
DATASET="MATH-500"
TASK_NAME="MATH-500"

cd /lambda/nfs/latense

# Phase 1: Vector Extraction
mkdir -p ./vectors
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    echo "--------------------------------------------------------"
    echo "Extracting vector for $MODEL on $DATASET..."
    python3 src/create_steering_vectors_cli.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --task_name "$TASK_NAME" \
        --num_samples 100 \
        --split "test[400:]" \
        --layer_idx -1
done

# Phase 2: Layer Sensitivity Sweep
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    echo "--------------------------------------------------------"
    echo "Starting Layer Sweep for $MODEL on $DATASET..."
    python3 src/run_layer_sensitivity.py \
        --model_name_or_path "$MODEL" \
        --dataset "$DATASET" \
        --split "test[200:400]" \
        --vector_path "./vectors/${TASK_NAME}_${MODEL_NAME}_L-1.pt" \
        --layers "20,24,28,31" \
        --num_samples $SWEEP_SAMPLES \
        --output_dir "./results_sweeps/layer_sensitivity_${MODEL_NAME}_math500"
done

# Phase 3: Alpha Hyper-Sweep
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    echo "--------------------------------------------------------"
    echo "Starting Alpha Sweep for $MODEL on $DATASET..."
    python3 src/run_alpha_sweep.py \
        --model_name_or_path "$MODEL" \
        --dataset "$DATASET" \
        --split "test[200:400]" \
        --vector_path "./vectors/${TASK_NAME}_${MODEL_NAME}_L-1.pt" \
        --layer_idx 28 \
        --alphas "0.1,0.3,0.5,0.7,1.0" \
        --num_samples $SWEEP_SAMPLES \
        --output_dir "./results_sweeps/alpha_sweep_${MODEL_NAME}_math500"
done

# Phase 4: Final Evaluation (3 Seeds)
SEEDS=(42 43 44)
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    for SEED in "${SEEDS[@]}"; do
        for MODE in "latense" "greedy_cot"; do
            echo "--------------------------------------------------------"
            echo "Final Eval: $MODEL | Mode: $MODE | Seed: $SEED | Dataset: $DATASET"
            python3 src/main.py \
                --model_name_or_path "$MODEL" \
                --dataset "$DATASET" \
                --split "test[:200]" \
                --generation_mode "$MODE" \
                --output_dir "./results_final/${MODEL_NAME}_${MODE}_seed${SEED}_math500" \
                --vector_name_template "./vectors/${TASK_NAME}_{model_name}_L-1.pt" \
                --layer_idx 28 \
                --alpha 0.3 \
                --max_eval_samples $LIMIT \
                --seed $SEED >> logs_final_eval_math500.log 2>&1
        done
    done
done

echo "MATH-500 PIPELINE COMPLETE."
