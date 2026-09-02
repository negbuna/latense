#!/bin/bash

# Usage: ./src/run_parallel_eval.sh <dataset> <model> <cluster_dir> <output_dir> [split] [limit] [alpha]
# Example: ./src/run_parallel_eval.sh "openai/gsm8k" "meta-llama/Llama-3.1-8B-Instruct" "./vectors/gsm8k_llama" "./results" "test" 500 0.3

DATASET=$1
MODEL=$2
CLUSTER_DIR=$3
OUTPUT_DIR=$4
SPLIT=${5:-test}
LIMIT=${6:-""}
ALPHA=${7:-0.3}

echo "========================================================"
echo "STARTING EVALUATION"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 python3 src/main.py \
    --model_name_or_path "$MODEL" \
    --dataset "$DATASET" \
    --generation_mode "latense" \
    --cluster_dir "$CLUSTER_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --seed 42 \
    ${LIMIT:+--limit $LIMIT} \
    --alpha $ALPHA

echo "Evaluation complete."