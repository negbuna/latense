#!/bin/bash

# run_static_h100_vllm.sh
# Establishing Static Steering (CAA) baseline for NeurIPS Table 1
# Optimized with vLLM for high-throughput generation

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-2-9b-it"
    "Qwen/Qwen2.5-7B-Instruct"
)

DATASETS=(
    "HuggingFaceH4/MATH-500"
    "wics/strategy-qa"
    "mandarjoshi/trivia_qa"
)

SEEDS=(42 43 44)

# Static Steering Hyperparams
ALPHA=0.3
LAYER_IDX=-1
VECTORS_DIR="./vectors"
RESULTS_DIR="./results_static"
LOG_DIR="./logs_static"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

push_results() {
    local PATH_TO_ADD=$1
    local MSG=$2
    
    echo "[$(date +%T)] Pushing results for: $MSG"
    git add "$PATH_TO_ADD"
    git commit -m "NeurIPS Static Baseline (vLLM): $MSG"
    
    local success=false
    for i in {1..5}; do
        if git pull --rebase origin data-snapshot && git push origin data-snapshot; then
            success=true
            break
        fi
        echo "[$(date +%T)] Push failed, retrying... ($i/5)"
        sleep 10
    done
}

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(basename "$MODEL")
    for DATASET in "${DATASETS[@]}"; do
        DATA_SHORT=${DATASET//\//_}
        VEC_DATA_PREFIX=$(echo "$DATA_SHORT" | sed 's/HuggingFaceH4_//; s/wics_//; s/mandarjoshi_//; s/_/-/g')
        
        SPLIT="test"
        LIMIT=500
        if [[ "$DATASET" == *"trivia_qa"* ]]; then
            SPLIT="validation"
        elif [[ "$DATASET" == *"MATH-500"* ]]; then
            SPLIT="test[:500]"
        fi

        # Find the vector path
        VEC_PATH="$VECTORS_DIR/${VEC_DATA_PREFIX}_${MODEL_SHORT}_L-1.pt"
        if [[ "$VEC_DATA_PREFIX" == "strategy-qa" ]]; then
             VEC_PATH="$VECTORS_DIR/strategy-qa_${MODEL_SHORT}_L-1.pt"
        fi
        if [ ! -f "$VEC_PATH" ]; then
            VEC_PATH=$(find "$VECTORS_DIR" -name "*${MODEL_SHORT}*_L-1.pt" | grep -i "${VEC_DATA_PREFIX/-/_}" | head -n 1)
        fi

        if [ ! -f "$VEC_PATH" ]; then
            echo "Warning: No vector found for $MODEL_SHORT on $VEC_DATA_PREFIX. Skipping."
            continue
        fi

        for SEED in "${SEEDS[@]}"; do
            echo "[$(date +%T)] Launching vLLM: Model=$MODEL_SHORT | Data=$VEC_DATA_PREFIX | Seed=$SEED"
            
            OUTPUT_PATH="$RESULTS_DIR/${MODEL_SHORT}_${VEC_DATA_PREFIX}_seed${SEED}_static"
            LOG_FILE="$LOG_DIR/${MODEL_SHORT}_${VEC_DATA_PREFIX}_seed${SEED}_vllm.log"

            python3 src/run_vllm_static.py \
                --model_name_or_path "$MODEL" \
                --dataset "$DATASET" \
                --vector_path "$VEC_PATH" \
                --output_dir "$OUTPUT_PATH" \
                --alpha "$ALPHA" \
                --layer_idx "$LAYER_IDX" \
                --seed "$SEED" \
                --split "$SPLIT" \
                --max_eval_samples "$LIMIT" \
                > "$LOG_FILE" 2>&1
            
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 0 ]; then
                echo "[$(date +%T)] Finished successfully: $MODEL_SHORT | $VEC_DATA_PREFIX | Seed=$SEED"
                push_results "$OUTPUT_PATH" "$MODEL_SHORT $VEC_DATA_PREFIX Seed $SEED"
            else
                echo "[$(date +%T)] FAILED (Exit $EXIT_CODE): $MODEL_SHORT | $VEC_DATA_PREFIX | Seed=$SEED"
            fi
        done
    done
done

echo "ALL 27 STATIC BASELINE EXPERIMENTS COMPLETE (vLLM)."
