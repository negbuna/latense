#!/bin/bash
set -e

# Configuration
DATASET="wics/strategy-qa"
VECTORS_DIR="./vectors"
RESULTS_DIR="./results_final"
mkdir -p $RESULTS_DIR

cd /lambda/nfs/latense

echo "STRATEGY-QA FINAL EVALUATION STARTED at $(date)"

SEEDS=(42 43 44)

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------------------"
    echo "SEED: $SEED"
    echo "----------------------------------------------------"

    # Llama
    echo "Final Eval: Llama-3.1-8B | Seed: $SEED"
    python3 src/main.py \
        --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
        --dataset "$DATASET" \
        --generation_mode "latense" \
        --layer_idx 24 \
        --alpha 0.3 \
        --seed $SEED \
        --split "test[:500]" \
        --vector_name_template "./vectors/strategy-qa_{model_name}_L-1.pt" \
        --output_dir "$RESULTS_DIR/llama_latense_seed$SEED" >> logs_final_eval.log 2>&1
    
    # Gemma
    echo "Final Eval: Gemma-2-9b-it | Seed: $SEED"
    python3 src/main.py \
        --model_name_or_path "google/gemma-2-9b-it" \
        --dataset "$DATASET" \
        --generation_mode "latense" \
        --layer_idx 20 \
        --alpha 0.3 \
        --seed $SEED \
        --split "test[:500]" \
        --vector_name_template "./vectors/strategy-qa_{model_name}_L-1.pt" \
        --output_dir "$RESULTS_DIR/gemma_latense_seed$SEED" >> logs_final_eval.log 2>&1
    
    # Qwen
    echo "Final Eval: Qwen2.5-7B-Instruct | Seed: $SEED"
    python3 src/main.py \
        --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
        --dataset "$DATASET" \
        --generation_mode "latense" \
        --layer_idx 26 \
        --alpha 0.3 \
        --seed $SEED \
        --split "test[:500]" \
        --vector_name_template "./vectors/strategy-qa_{model_name}_L-1.pt" \
        --output_dir "$RESULTS_DIR/qwen_latense_seed$SEED" >> logs_final_eval.log 2>&1
done

echo "STRATEGY-QA FINAL EVALUATION COMPLETED at $(date)"

# --- Transition to Baselines for MATH-500 and TriviaQA ---
echo "STARTING MASTER BASELINES (MATH-500 & TriviaQA) at $(date)"
bash src/run_all_baselines_master.sh

# --- Transition to MATH-500 Pipeline ---
echo "STARTING MATH-500 PIPELINE at $(date)"
bash src/run_math500_pipeline.sh >> logs_math500_master.log 2>&1
