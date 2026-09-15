#!/bin/bash

# Usage: ./src/run_vector_pipeline.sh <dataset_name> <model_path> <n_clusters> <output_base_dir> [threshold] [max_samples] [seed] [split]
# Example: ./src/run_vector_pipeline.sh "openai/gsm8k" "meta-llama/Llama-3.1-8B-Instruct" 3 "./vectors/gsm8k_llama" 3 2000 42 "train[200:]"

DATASET=$1
MODEL=$2
N_CLUSTERS=$3
OUTPUT_BASE=$4
THRESH=${5:-4}  # default to 4 if not provided
MAX_SAMPLES=${6:-""} # optional max samples
SEED=${7:-42}
SPLIT=${8:-train}

set -e  # exit on error

echo "========================================================"
echo "STARTING VECTOR PIPELINE"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Clusters: $N_CLUSTERS"
echo "Output: $OUTPUT_BASE"
echo "Threshold: $THRESH"
echo "Max Samples: ${MAX_SAMPLES:-All}"
echo "Seed: $SEED"
echo "Split: $SPLIT"
echo "========================================================"

# 1. CLUSTER VECTORS
echo "[1/3] Clustering Training Data..."
python src/cluster_vectors.py \
    --dataset "$DATASET" \
    --model_name_or_path "$MODEL" \
    --n_clusters $N_CLUSTERS \
    --output_dir "$OUTPUT_BASE" \
    --split "$SPLIT" \
    ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES} \
    --seed $SEED

# loop through each cluster to collect states and compute vectors
for (( i=0; i<$N_CLUSTERS; i++ ))
do
    echo "--------------------------------------------------------"
    echo "Processing Cluster $i"
    
    INDICES_FILE="$OUTPUT_BASE/cluster_${i}_indices.txt"
    CLUSTER_STATE_DIR="$OUTPUT_BASE/states_cluster_$i"
    
    # 2. COLLECT STATES
    echo "[2/3] Collecting States for Cluster $i..."
    python src/collect_states.py \
        --model_name_or_path "$MODEL" \
        --dataset "$DATASET" \
        --indices_file "$INDICES_FILE" \
        --output_dir "$CLUSTER_STATE_DIR" \
        --split "$SPLIT" \
        --layer_idx -1 \
        --solver_prompt_idx 0 \
        --good_example_threshold $THRESH \
        --seed $SEED

    # Construct the path where collect_states.py saved the files
    # Note: collect_states appends "{model}-{dataset}/state_collection/layer{layer}_prompt{prompt}[_thresh{thresh}]/hidden_states/"
    # We need to find this directory dynamically or construct it.
    # Assuming default prompt_idx=0, thresh=4, layer_idx=-1
    MODEL_NAME=$(basename "$MODEL")
    DATA_NAME=${DATASET//\//-}
    
    if [[ "$DATASET" == *"strategy"* ]] || [[ "$DATASET" == *"trivia_qa"* ]]; then
        DIR_SUFFIX="layer-1_prompt0"
    else
        DIR_SUFFIX="layer-1_prompt0_thresh${THRESH}"
    fi

    STATES_PATH="$CLUSTER_STATE_DIR/$MODEL_NAME-$DATA_NAME/state_collection/$DIR_SUFFIX/hidden_states/"

    # 3. COMPUTE VECTOR
    echo "[3/3] Computing Vector for Cluster $i..."
    python src/compute_steering_vector.py \
        --states_dir "$STATES_PATH" \
        --output_file "$OUTPUT_BASE/vector_$i.pt"
        
    echo "Cluster $i Complete. Vector saved to $OUTPUT_BASE/vector_$i.pt"
done

echo "========================================================"
echo "PIPELINE COMPLETE"
echo "Centroids and Vectors are located in: $OUTPUT_BASE"
echo "========================================================"