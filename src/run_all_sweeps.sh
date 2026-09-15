#!/bin/bash

# Usage: ./src/run_all_sweeps.sh [seed]
# Runs alpha sweeps for all configurations in parallel using available GPUs.
# Prerequisite: You must have run Phase 2 (Vector Creation) of run_experiment_8gpu.sh first!

SEED=${1:-42}

# Define the same configs as run_experiment_8gpu.sh
# Format: "Dataset|Model|Threshold|MaxSamples|VectorSplit|SweepSplit|EvalSplit|EvalLimit|Alpha"
CONFIGS=(
    "openai/gsm8k|meta-llama/Llama-3.1-8B-Instruct|3|1000|train[200:]|train[:200]|test|500|0.3"
    "wics/strategy-qa|meta-llama/Llama-3.1-8B-Instruct|3||test[1200:]|test[1000:1200]|test[:500]|500|0.3"
    "mandarjoshi/trivia_qa|meta-llama/Llama-3.1-8B-Instruct|3|1000|train[200:]|train[:200]|validation|500|0.3"
    "openai/gsm8k|Qwen/Qwen2.5-7B-Instruct|4|1000|train[200:]|train[:200]|test|500|0.3"
    "wics/strategy-qa|Qwen/Qwen2.5-7B-Instruct|3||test[1200:]|test[1000:1200]|test[:500]|500|0.3"
    "mandarjoshi/trivia_qa|Qwen/Qwen2.5-7B-Instruct|3|1000|train[200:]|train[:200]|validation|500|0.3"
    "openai/gsm8k|google/gemma-2-9b-it|4|1000|train[200:]|train[:200]|test|500|0.3"
    "wics/strategy-qa|google/gemma-2-9b-it|3||test[1200:]|test[1000:1200]|test[:500]|500|0.3"
    "mandarjoshi/trivia_qa|google/gemma-2-9b-it|3|1000|train[200:]|train[:200]|validation|500|0.3"
    "openai/gsm8k|google/gemma-2-9b-it|3|1000|train[200:]|train[:200]|test|500|0.3"
)

VECTORS_DIR="./vectors"
LOG_DIR="./logs/sweeps"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/alpha_setting_*.sh # Clear old settings from previous runs

# Safety check
if [ ! -d "./src" ]; then
    echo "Error: Run from project root."
    exit 1
fi

run_sweep() {
    local GPU_ID=$1
    local CONFIG_STR=$2
    
    IFS='|' read -r DATASET MODEL THRESH MAX_SAMPLES VECTOR_SPLIT SWEEP_SPLIT EVAL_SPLIT EVAL_LIMIT ALPHA <<< "$CONFIG_STR"
    
    local MODEL_SHORT=$(basename "$MODEL")
    local DATA_SHORT=${DATASET//\//_}
    local VEC_PATH="$VECTORS_DIR/${DATA_SHORT}_${MODEL_SHORT}_thresh${THRESH}_seed${SEED}"
    local LOG_FILE="$LOG_DIR/SWEEP_${DATA_SHORT}_${MODEL_SHORT}_thresh${THRESH}.log"

    if [ ! -d "$VEC_PATH" ]; then
        echo "[GPU $GPU_ID] Skipping $MODEL on $DATASET (Vectors not found at $VEC_PATH)"
        return
    fi

    echo "[GPU $GPU_ID] Sweeping $MODEL on $DATASET (Logs: $LOG_FILE)..."
    
    # Run the sweep script
    # We capture stdout to log file
    ./src/run_alpha_sweep.sh "$DATASET" "$MODEL" "$VEC_PATH" "$GPU_ID" "$SWEEP_SPLIT" > "$LOG_FILE" 2>&1
    
    # Extract recommendation for display
    local BEST_ALPHA=$(grep "Best Alpha:" "$LOG_FILE" | awk '{print $3}')
    local BEST_ACC=$(grep "Best Alpha:" "$LOG_FILE" | awk -F'Accuracy: ' '{print $2}' | tr -d ')')
    
    echo "[GPU $GPU_ID] Finished $MODEL on $DATASET. Best Alpha: $BEST_ALPHA (Acc: $BEST_ACC)"

    # Save best alpha for automation
    if [ ! -z "$BEST_ALPHA" ]; then
        local SAFE_DATA=${DATA_SHORT//-/_}
        local SAFE_MODEL=${MODEL_SHORT//-/_}
        SAFE_MODEL=${SAFE_MODEL//./_}
        local VAR_NAME="ALPHA_${SAFE_DATA}_${SAFE_MODEL}_${THRESH}"
        echo "export ${VAR_NAME}=${BEST_ALPHA}" > "$LOG_DIR/alpha_setting_${DATA_SHORT}_${MODEL_SHORT}_${THRESH}.sh"
    fi
}

echo "========================================"
echo "STARTING ALPHA SWEEPS (DEV SPLIT)"
echo "========================================"

# Launch first 8 tasks
for i in {0..7}; do
    run_sweep $i "${CONFIGS[$i]}" &
done
wait

# Launch remaining tasks (Indices 8 and 9)
run_sweep 0 "${CONFIGS[8]}" &
run_sweep 1 "${CONFIGS[9]}" &
wait

echo "========================================"
echo "SWEEP RESULTS SUMMARY"
echo "========================================"
echo "Check these values and update src/run_experiment_8gpu.sh:"
echo ""

for log in $LOG_DIR/*.log; do
    # Extract info from filename and content
    FILENAME=$(basename "$log")
    # Format: SWEEP_dataset_model_thresh.log
    # Just grep the best alpha line
    BEST_INFO=$(grep "Best Alpha:" "$log")
    if [ ! -z "$BEST_INFO" ]; then
        echo "$FILENAME -> $BEST_INFO"
    else
        echo "$FILENAME -> Failed (Check logs)"
    fi
done

# Aggregate best alphas into a single sourceable file
echo "# Auto-generated best alphas from run_all_sweeps.sh" > ./src/best_alphas.sh
cat "$LOG_DIR"/alpha_setting_*.sh >> ./src/best_alphas.sh 2>/dev/null || true
echo "Saved best alphas to ./src/best_alphas.sh"

echo "========================================"