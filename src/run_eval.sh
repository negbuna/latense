#!/bin/bash

# Usage: ./src/run_eval.sh <dataset_name> <model_path> <cluster_dir> <output_dir>
# Example: ./src/run_eval.sh "wics/strategy_qa" "google/gemma-2-9b-it" "./vectors/strategyqa_gemma" "./results"

DATASET=$1
MODEL=$2
CLUSTER_DIR=$3
OUTPUT_DIR=$4

echo "========================================================"
echo "STARTING EVALUATION"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Cluster Dir: $CLUSTER_DIR"
echo "Output Dir: $OUTPUT_DIR"
echo "========================================================"

python src/main.py \
    --model_name_or_path "$MODEL" \
    --dataset "$DATASET" \
    --generation_mode "latense" \
    --cluster_dir "$CLUSTER_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --verbose True