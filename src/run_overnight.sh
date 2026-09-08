#!/bin/bash

# Usage: ./src/run_overnight.sh
# Executes the heavy compute phases (Baselines + Vector Creation) for all seeds.

echo "Starting Overnight Run..."

# 1. Run Baselines (Only need to do this once)
echo "Phase 1: Baselines"
./src/run_experiment_8gpu.sh baselines

# 2. Create Vectors for all 3 seeds
echo "Phase 2: Vectors (Seed 42)"
./src/run_experiment_8gpu.sh vectors 42

echo "Phase 2: Vectors (Seed 43)"
./src/run_experiment_8gpu.sh vectors 43

echo "Phase 2: Vectors (Seed 44)"
./src/run_experiment_8gpu.sh vectors 44

echo "Overnight run complete. Ready for Alpha Sweeps."