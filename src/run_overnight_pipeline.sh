#!/bin/bash

# Configuration
LIMIT=500
SWEEP_SAMPLES=50
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it" "Qwen/Qwen2.5-7B-Instruct")
DATASETS=("wics/strategy-qa" "MATH-500")

# Wait for the current baseline run (PID 6265) to finish
echo "Waiting for PID 6265 (Baselines) to complete..."
while kill -0 6265 2>/dev/null; do
    sleep 60
done
echo "Baselines finished. Starting Overnight Pipeline..."

# Phase 1: Vector Extraction
mkdir -p ./vectors
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        TASK_NAME=$(echo "$DATASET" | cut -d'/' -f2)
        echo "Extracting vector for $MODEL on $DATASET..."
        python3 src/create_steering_vectors_cli.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --task_name "$TASK_NAME" \
            --num_samples 200 \
            --layer_idx -1
    done
done

# Phase 2: Layer Sensitivity Sweep
# We test layers 20, 24, 28, 31 to find the reasoning geometry
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    echo "Starting Layer Sweep for $MODEL..."
    python3 src/run_layer_sensitivity.py \
        --model_name_or_path "$MODEL" \
        --dataset "wics/strategy-qa" \
        --vector_path "./vectors/strategy-qa_${MODEL_NAME}_L-1.pt" \
        --layers "20,24,28,31" \
        --num_samples $SWEEP_SAMPLES \
        --output_dir "./results_sweeps/layer_sensitivity_${MODEL_NAME}"
done

# Phase 3: Alpha Hyper-Sweep
# Testing strengths: 0.1, 0.3, 0.5, 0.7, 1.0
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    # Using layer 28 as a default "high-reasoning" layer if sweep isn't processed yet
    echo "Starting Alpha Sweep for $MODEL..."
    python3 src/run_alpha_sweep.py \
        --model_name_or_path "$MODEL" \
        --dataset "wics/strategy-qa" \
        --vector_path "./vectors/strategy-qa_${MODEL_NAME}_L-1.pt" \
        --layer_idx 28 \
        --alphas "0.1,0.3,0.5,0.7,1.0" \
        --num_samples $SWEEP_SAMPLES \
        --output_dir "./results_sweeps/alpha_sweep_${MODEL_NAME}"
done

# Phase 4: Final Comparison (Multi-Seed for Statistical Significance)
# Comparing full LaTense vs Static Steering vs Baseline
SEEDS=(42 43 44)
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    for SEED in "${SEEDS[@]}"; do
        for MODE in "latense" "greedy_cot"; do
            echo "Final Eval: $MODEL | Mode: $MODE | Seed: $SEED"
            python3 src/main.py \
                --model_name_or_path "$MODEL" \
                --dataset "wics/strategy-qa" \
                --generation_mode "$MODE" \
                --output_dir "./results_final/${MODEL_NAME}_${MODE}_seed${SEED}" \
                --vector_path "./vectors/strategy-qa_${MODEL_NAME}_L-1.pt" \
                --layer_idx 28 \
                --alpha 0.5 \
                --max_eval_samples $LIMIT \
                --seed $SEED
        done
    done
done

echo "OVERNIGHT PIPELINE COMPLETE."
