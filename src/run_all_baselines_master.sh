#!/bin/bash
set -e

MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it" "Qwen/Qwen2.5-7B-Instruct")
BASE_RESULTS_DIR="./results_full_baselines"
mkdir -p $BASE_RESULTS_DIR

# 1. MATH-500 Baselines
DATASET="MATH-500"
SPLIT="test[:200]" # Following disjointness: test[:200] for final eval/baselines
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    for MODE in "greedy_cot" "self_consistency"; do
        echo "Running MATH-500 Baseline: $MODEL_NAME | $MODE"
        python3 src/main.py \
            --model_name_or_path "$MODEL" \
            --dataset "$DATASET" \
            --generation_mode "$MODE" \
            --output_dir "$BASE_RESULTS_DIR/${MODEL_NAME}_${MODE}_math500" \
            --split "$SPLIT" \
            --sc_k 5 \
            --seed 42 >> logs_baselines_math500.log 2>&1
    done
done

# 2. TriviaQA Baselines
DATASET="trivia_qa"
SPLIT="validation[:500]" # Following disjointness: validation[:500] for final eval/baselines
for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(echo "$MODEL" | cut -d'/' -f2)
    for MODE in "greedy_cot" "self_consistency"; do
        echo "Running TriviaQA Baseline: $MODEL_NAME | $MODE"
        python3 src/main.py \
            --model_name_or_path "$MODEL" \
            --dataset "$DATASET" \
            --generation_mode "$MODE" \
            --output_dir "$BASE_RESULTS_DIR/${MODEL_NAME}_${MODE}_triviaqa" \
            --split "$SPLIT" \
            --sc_k 5 \
            --seed 42 >> logs_baselines_triviaqa.log 2>&1
    done
done

echo "ALL BASELINES COMPLETED at $(date)"
