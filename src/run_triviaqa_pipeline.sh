#!/bin/bash
set -e

# Configuration
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it" "Qwen/Qwen2.5-7B-Instruct")
DATASET="trivia_qa"
TASK_NAME="trivia_qa"
RESULTS_DIR="./results_final_neurips"
SWEEP_DIR="./results_sweeps/triviaqa"
mkdir -p $RESULTS_DIR $SWEEP_DIR

cd /lambda/nfs/latense

echo "TRIVIA-QA PIPELINE STARTED at $(date)"

# Phase 1: Vector Extraction
for MODEL in "${MODELS[@]}"; do
    echo "Extracting TriviaQA vector for $MODEL..."
    python3 src/create_steering_vectors_cli.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --task_name "$TASK_NAME" \
        --num_samples 500 \
        --split "train[200:]" \
        --layer_idx -1
done

# Phase 2: Sweeps
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    echo "Running Sweeps for $MODEL..."
    python3 src/run_layer_sensitivity.py \
        --model_name_or_path "$MODEL" \
        --dataset "$DATASET" \
        --split "train[:200]" \
        --vector_path "./vectors/${TASK_NAME}_${MODEL_NAME}_L-1.pt" \
        --layers "16,20,24,28,31" \
        --num_samples 100 \
        --output_dir "$SWEEP_DIR/layer_${MODEL_NAME}"
        
    python3 src/run_alpha_sweep.py \
        --model_name_or_path "$MODEL" \
        --dataset "$DATASET" \
        --split "train[:200]" \
        --vector_path "./vectors/${TASK_NAME}_${MODEL_NAME}_L-1.pt" \
        --layer_idx 24 \
        --alphas "0.1,0.3,0.5,0.7,1.0" \
        --num_samples 100 \
        --output_dir "$SWEEP_DIR/alpha_${MODEL_NAME}"
done

# Phase 3: Final Eval (3 Seeds)
# Note: Using optimized Alpha 0.1 for TriviaQA as a safe baseline based on Math results
SEEDS=(42 43 44)
for SEED in "${SEEDS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
        for MODE in "latense" "greedy_cot"; do
            echo "Final Eval TriviaQA: $MODEL_NAME | Mode: $MODE | Seed: $SEED"
            python3 src/main.py \
                --model_name_or_path "$MODEL" \
                --dataset "$DATASET" \
                --generation_mode "$MODE" \
                --layer_idx 24 \
                --alpha 0.1 \
                --seed $SEED \
                --do_sample \
                --temperature 0.7 \
                --split "validation[:500]" \
                --vector_name_template "./vectors/trivia_qa_{model_name}_L-1.pt" \
                --output_dir "$RESULTS_DIR/${MODEL_NAME}_${MODE}_seed${SEED}_triviaqa" >> logs_neurips_trivia.log 2>&1
        done
    done
done

echo "TRIVIA-QA PIPELINE COMPLETED at $(date)"
