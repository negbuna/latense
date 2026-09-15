#!/bin/bash

# Usage: ./src/run_baselines.sh <dataset_name> <model_path> <output_dir> <mode> [split] [limit]
# mode options: "greedy_cot" or "self_consistency"
# Example: ./src/run_baselines.sh "wics/strategy_qa" "google/gemma-2-9b-it" "./results" "greedy_cot" "dev" 500

DATASET=$1
MODEL=$2
OUTPUT_DIR=$3
MODE=$4
SPLIT=${5:-test}  # Default to test if not provided
LIMIT=${6:-""}

echo "========================================================"
echo "STARTING BASELINE: $MODE"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 python3 src/main.py \
    --model_name_or_path "$MODEL" \
    --dataset "$DATASET" \
    --generation_mode "$MODE" \
    --output_dir "$OUTPUT_DIR" \
    --sc_k 5 \
    --verbose True \
    --split "$SPLIT" \
    --seed 42 \
    ${LIMIT:+--limit $LIMIT}