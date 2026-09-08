#!/bin/bash

set -e

echo "Checking disk space..."
df -h .

# Remove system package that breaks pip (flatbuffers)
echo "Removing conflicting system packages..."
sudo apt-get remove -y python3-flatbuffers || true

# 0. Aggressive Cleanup (Fixes corrupted installations)
echo "Cleaning up corrupted packages..."
pip uninstall -y transformers tokenizers sentence-transformers accelerate numpy scipy huggingface-hub safetensors protobuf latex2sympy2 math_verify antlr4-python3-runtime tf-keras datasets pyarrow || true
rm -rf /home/ubuntu/.local/lib/python3.10/site-packages/transformers*
rm -rf /home/ubuntu/.local/lib/python3.10/site-packages/tokenizers*
rm -rf /home/ubuntu/.local/lib/python3.10/site-packages/numpy*
rm -rf /home/ubuntu/.local/lib/python3.10/site-packages/scipy*
rm -rf /home/ubuntu/.local/lib/python3.10/site-packages/huggingface_hub*
rm -rf /home/ubuntu/.local/lib/python3.10/site-packages/safetensors*
rm -rf /home/ubuntu/.local/lib/python3.10/site-packages/latex2sympy2*
rm -f /home/ubuntu/.local/lib/python3.10/site-packages/latex2sympy2.py
rm -rf /home/ubuntu/.local/lib/python3.10/site-packages/math_verify*
rm -rf /home/ubuntu/.local/lib/python3.10/site-packages/antlr4*

# 1. Install Dependencies
echo "Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install --upgrade --no-cache-dir -r requirements.txt
echo "Installing local latex2sympy..."
pip install ./src/extract_judge_answer/latex2sympy

# 2. Verify Imports
echo ""
echo "Verifying environment..."
python -c "import torch; print(f'Torch {torch.__version__} is working.'); import transformers; from transformers import PreTrainedModel; print(f'Success! Transformers {transformers.__version__} is working.')"

# 3. Hugging Face Login
echo ""
echo "=================================================================="
echo "HUGGING FACE LOGIN REQUIRED"
echo "You need access to Llama-3.1 and Gemma-2."
echo "1. Get your token from: https://huggingface.co/settings/tokens"
echo "2. Run this command manually:"
echo "   huggingface-cli login"
echo "=================================================================="

# 4. Pre-download Models
echo ""
echo "Pre-downloading model weights to cache..."
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
models = ['meta-llama/Llama-3.1-8B-Instruct', 'Qwen/Qwen2.5-7B-Instruct', 'google/gemma-2-9b-it']
for m in models:
    print(f'Downloading {m}...')
    try: AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)
    except Exception as e: print(f'Note: Could not fully download {m} (might need login): {e}')

python src/download_datasets.py
"

chmod +x src/*.sh