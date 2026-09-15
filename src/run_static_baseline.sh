#!/bin/bash
# Budget-Safe Static Steering Baseline Runner
# Established for NeurIPS 2026 Rebuttal
# Estimated Runtime: ~18 hours on 1x A100 | ~4.5 hours on 4x A100

MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it" "Qwen/Qwen2.5-7B-Instruct")
DATASETS=("wics/strategy-qa" "MATH-500" "mandarjoshi/trivia_qa")
SEEDS=(42 43 44)
VECTORS_DIR="./vectors"
RESULTS_DIR="./results_static"

mkdir -p $RESULTS_DIR

echo "========================================================"
echo "STARTING STATIC STEERING (CAA) BASELINE RUNS"
echo "Target: results_static/"
echo "========================================================"

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename $MODEL)
    for DATASET in "${DATASETS[@]}"; do
        # Mapping dataset name for vector filenames
        DATA_KEY=$DATASET
        if [[ "$DATASET" == "wics/strategy-qa" ]]; then DATA_KEY="strategy-qa"; fi
        if [[ "$DATASET" == "mandarjoshi/trivia_qa" ]]; then DATA_KEY="trivia_qa"; fi
        
        # Identify pre-computed vector
        # Template: ./vectors/{dataset}_{model}_L-1.pt
        VECTOR_PATH="$VECTORS_DIR/${DATA_KEY}_${MODEL_NAME}_L-1.pt"

        if [[ ! -f "$VECTOR_PATH" ]]; then
            echo "ERROR: Vector not found at $VECTOR_PATH. Skipping."
            continue
        fi

        for SEED in "${SEEDS[@]}"; do
            echo "[$(date)] RUNNING: $MODEL_NAME | $DATASET | Seed $SEED"
            
            # MATH-500 uses 200 samples, others use 500
            LIMIT=500
            if [[ "$DATASET" == "MATH-500" ]]; then LIMIT=200; fi

            python src/main.py \
                --model_name_or_path "$MODEL" \
                --dataset "$DATASET" \
                --output_dir "$RESULTS_DIR/${MODEL_NAME}_${DATA_KEY}_seed${SEED}" \
                --max_eval_samples $LIMIT \
                --generation_mode "latense" \
                --vector_name_template "$VECTOR_PATH" \
                --alpha 0.3 \
                --no_dynamic_scaling \
                --no_similarity_modulation \
                --seed $SEED \
                --verbose False
        done
    done
done

echo "========================================================"
echo "ALL STATIC BASELINES COMPLETE"
echo "========================================================"
