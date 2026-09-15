#!/bin/bash

# run_static_h100.sh
# Establishing Static Steering (CAA) baseline for NeurIPS Table 1
# Optimized for H100 80GB: 4 Parallel processes + FlashAttention-2

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
    while ! mkdir .git_push_lock 2>/dev/null; do sleep $((RANDOM % 10 + 5)); done
    git add "$PATH_TO_ADD"
    git commit -m "NeurIPS Static Baseline: $MSG"
    local success=false
    for i in {1..5}; do
        if git pull --rebase origin data-snapshot && git push origin data-snapshot; then
            success=true; break
        fi
        sleep 10
    done
    rmdir .git_push_lock
}

# Balanced Scheduling: Interleave datasets to mix heavy/light tasks
MAX_PARALLEL=4

# Create a flattened list of tasks
tasks=()
for SEED in "${SEEDS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            tasks+=("$MODEL|$DATASET|$SEED")
        done
    done
done

for task in "${tasks[@]}"; do
    IFS='|' read -r MODEL DATASET SEED <<< "$task"
    
    while [ $(jobs -r | wc -l) -ge $MAX_PARALLEL ]; do
        sleep 5
    done

    MODEL_SHORT=$(basename "$MODEL")
    DATA_SHORT=${DATASET//\//_}
    VEC_DATA_PREFIX=$(echo "$DATA_SHORT" | sed 's/HuggingFaceH4_//; s/wics_//; s/mandarjoshi_//; s/_/-/g')
    
    SPLIT="test"
    LIMIT=500
    [[ "$DATASET" == *"trivia_qa"* ]] && SPLIT="validation"
    [[ "$DATASET" == *"MATH-500"* ]] && SPLIT="test[:500]"
    
    # Deadline Mitigation: Cap Gemma-2 on MATH-500 to 300 samples
    if [[ "$MODEL" == *"gemma-2"* && "$DATASET" == *"MATH-500"* ]]; then
        LIMIT=300
    fi

    VEC_PATH="$VECTORS_DIR/${VEC_DATA_PREFIX}_${MODEL_SHORT}_L-1.pt"
    [[ "$VEC_DATA_PREFIX" == "strategy-qa" ]] && VEC_PATH="$VECTORS_DIR/strategy-qa_${MODEL_SHORT}_L-1.pt"
    if [ ! -f "$VEC_PATH" ]; then
        VEC_PATH=$(find "$VECTORS_DIR" -name "*${MODEL_SHORT}*_L-1.pt" | grep -i "${VEC_DATA_PREFIX/-/_}" | head -n 1)
    fi

    [ ! -f "$VEC_PATH" ] && continue

    OUTPUT_PATH="$RESULTS_DIR/${MODEL_SHORT}_${VEC_DATA_PREFIX}_seed${SEED}_static"
    LOG_FILE="$LOG_DIR/${MODEL_SHORT}_${VEC_DATA_PREFIX}_seed${SEED}.log"

    echo "[$(date +%T)] Launching: Model=$MODEL_SHORT | Data=$VEC_DATA_PREFIX | Seed=$SEED"
    (
        python3 src/main.py \
            --model_name_or_path "$MODEL" \
            --dataset "$DATASET" \
            --output_dir "$OUTPUT_PATH" \
            --split "$SPLIT" \
            --max_eval_samples "$LIMIT" \
            --generation_mode "latense" \
            --vector_name_template "$VEC_PATH" \
            --alpha "$ALPHA" \
            --seed "$SEED" \
            --no_dynamic_scaling \
            --no_similarity_modulation \
            --prompts_to_run "0" \
            --resume \
            > "$LOG_FILE" 2>&1
        
        [ $? -eq 0 ] && push_results "$OUTPUT_PATH" "$MODEL_SHORT $VEC_DATA_PREFIX Seed $SEED"
    ) &
    sleep 30
done

wait
echo "ALL EXPERIMENTS COMPLETE."
