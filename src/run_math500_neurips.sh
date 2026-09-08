#!/bin/bash
set -e

# Configuration
DATASET="MATH-500"
RESULTS_DIR="./results_final_neurips"
mkdir -p $RESULTS_DIR

cd /lambda/nfs/latense

echo "MATH-500 NEURIPS OPTIMIZED EVALUATION STARTED at $(date)"
echo "Mode: Sampling (T=0.7) | Seeds: 42, 43, 44"

# Models and their optimal parameters (found from sweeps)
# Llama-3.1-8B: Layer 28, Alpha 0.1
# Gemma-2-9b: Layer 28, Alpha 0.3
# Qwen2.5-7B: Layer 20, Alpha 0.1

SEEDS=(42 43 44)

for SEED in "${SEEDS[@]}"; do
    echo "----------------------------------------------------"
    echo "SEED: $SEED"
    echo "----------------------------------------------------"

    # 1. Llama-3.1-8B-Instruct
    MODEL="meta-llama/Llama-3.1-8B-Instruct"
    MODEL_NAME="Llama-3.1-8B-Instruct"
    for MODE in "latense" "greedy_cot"; do
        echo "Eval: $MODEL_NAME | Mode: $MODE | Seed: $SEED"
        python3 src/main.py \
            --model_name_or_path "$MODEL" \
            --dataset "$DATASET" \
            --generation_mode "$MODE" \
            --layer_idx 28 \
            --alpha 0.1 \
            --seed $SEED \
            --do_sample \
            --temperature 0.7 \
            --split "test[:200]" \
            --vector_name_template "./vectors/MATH-500_{model_name}_L-1.pt" \
            --output_dir "$RESULTS_DIR/${MODEL_NAME}_${MODE}_seed${SEED}" >> logs_neurips_math.log 2>&1
    done

    # 2. google/gemma-2-9b-it
    MODEL="google/gemma-2-9b-it"
    MODEL_NAME="gemma-2-9b-it"
    for MODE in "latense" "greedy_cot"; do
        echo "Eval: $MODEL_NAME | Mode: $MODE | Seed: $SEED"
        python3 src/main.py \
            --model_name_or_path "$MODEL" \
            --dataset "$DATASET" \
            --generation_mode "$MODE" \
            --layer_idx 28 \
            --alpha 0.3 \
            --seed $SEED \
            --do_sample \
            --temperature 0.7 \
            --split "test[:200]" \
            --vector_name_template "./vectors/MATH-500_{model_name}_L-1.pt" \
            --output_dir "$RESULTS_DIR/${MODEL_NAME}_${MODE}_seed${SEED}" >> logs_neurips_math.log 2>&1
    done

    # 3. Qwen/Qwen2.5-7B-Instruct
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    MODEL_NAME="Qwen2.5-7B-Instruct"
    for MODE in "latense" "greedy_cot"; do
        echo "Eval: $MODEL_NAME | Mode: $MODE | Seed: $SEED"
        python3 src/main.py \
            --model_name_or_path "$MODEL" \
            --dataset "$DATASET" \
            --generation_mode "$MODE" \
            --layer_idx 20 \
            --alpha 0.1 \
            --seed $SEED \
            --do_sample \
            --temperature 0.7 \
            --split "test[:200]" \
            --vector_name_template "./vectors/MATH-500_{model_name}_L-1.pt" \
            --output_dir "$RESULTS_DIR/${MODEL_NAME}_${MODE}_seed${SEED}" >> logs_neurips_math.log 2>&1
    done
done

echo "MATH-500 NEURIPS EVALUATION COMPLETED at $(date)"
echo 'Starting TriviaQA Phase...'
bash src/run_triviaqa_pipeline.sh >> logs_orchestrator.out 2>&1
