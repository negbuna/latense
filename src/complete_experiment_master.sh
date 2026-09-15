#!/bin/bash
set -e

# Configuration
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "google/gemma-2-9b-it" "Qwen/Qwen2.5-7B-Instruct")
DATASET="wics/strategy-qa"
VECTORS_DIR="./vectors"
RESULTS_DIR="./results_final"
mkdir -p $RESULTS_DIR

echo "MASTER EXPERIMENT SCRIPT STARTED at $(date)"

## --- Phase 1: Qwen Layer Sweep (Restart) ---
#echo "Starting Qwen Layer Sweep..."
#python3 src/run_layer_sensitivity.py \
#    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
#    --dataset "$DATASET" \
#    --vector_path "$VECTORS_DIR/strategy-qa_Qwen2.5-7B-Instruct_L-1.pt" \
#    --layers "20,22,24,26,28,30,31" \
#    --num_samples 50 \
#    --alpha 0.3 \
#    --max_new_tokens 256 > logs_sweep_qwen_layer_master.log 2>&1

## --- Phase 2: Gemma Alpha Sweep (L20 was optimal) ---
#echo "Starting Gemma Alpha Sweep (Layer 20)..."
#python3 src/run_alpha_sweep.py \
#    --model_name_or_path "google/gemma-2-9b-it" \
#    --dataset "$DATASET" \
#    --vector_path "$VECTORS_DIR/strategy-qa_gemma-2-9b-it_L-1.pt" \
#    --layer_idx 20 \
#    --alphas "0.1,0.3,0.5,0.7,1.0" \
#    --num_samples 50 > logs_sweep_gemma_alpha_master.log 2>&1

## --- Phase 3: Qwen Alpha Sweep (Determining Best Layer from Sweep) ---
## We'll use Layer 26 as a fallback/default reasoning layer if extraction is complex
#echo "Starting Qwen Alpha Sweep (Layer 26)..."
#python3 src/run_alpha_sweep.py \
#    --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" \
#    --dataset "$DATASET" \
#    --vector_path "$VECTORS_DIR/strategy-qa_Qwen2.5-7B-Instruct_L-1.pt" \
#    --layer_idx 26 \
#    --alphas "0.1,0.3,0.5,0.7,1.0" \
#    --num_samples 50 > logs_sweep_qwen_alpha_master.log 2>&1

# --- Phase 4: Final Evaluation (3 Seeds) ---
# Llama: L24, A0.3 | Gemma: L20, A0.3 (estimated) | Qwen: L26, A0.3 (estimated)
SEEDS=(42 43 44)
for SEED in "${SEEDS[@]}"; do
    # Llama
    echo "Final Eval: Llama-3.1-8B | Seed: $SEED"
    python3 src/main.py --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" --dataset "$DATASET" --generation_mode "latense" --layer_idx 24 --alpha 0.3 --seed $SEED --max_eval_samples 500 --output_dir "$RESULTS_DIR/llama_latense_seed$SEED" >> logs_final_eval.log 2>&1
    
    # Gemma
    echo "Final Eval: Gemma-2-9b-it | Seed: $SEED"
    python3 src/main.py --model_name_or_path "google/gemma-2-9b-it" --dataset "$DATASET" --generation_mode "latense" --layer_idx 20 --alpha 0.3 --seed $SEED --max_eval_samples 500 --output_dir "$RESULTS_DIR/gemma_latense_seed$SEED" >> logs_final_eval.log 2>&1
    
    # Qwen
    echo "Final Eval: Qwen2.5-7B-Instruct | Seed: $SEED"
    python3 src/main.py --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --dataset "$DATASET" --generation_mode "latense" --layer_idx 26 --alpha 0.3 --seed $SEED --max_eval_samples 500 --output_dir "$RESULTS_DIR/qwen_latense_seed$SEED" >> logs_final_eval.log 2>&1
done

echo "MASTER EXPERIMENT SCRIPT COMPLETED at $(date)"
