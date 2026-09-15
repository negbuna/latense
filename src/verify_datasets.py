from datasets import load_dataset
import sys

print("--- Verifying Dataset Loading ---")

try:
    print("1. Checking StrategyQA (test split)...")
    # This confirms we can load the 'test' split which replaces 'train'
    ds = load_dataset("wics/strategy-qa", split="test")
    print(f"   Success! Loaded {len(ds)} samples.")
except Exception as e:
    print(f"   FAILED: {e}")

try:
    print("2. Checking TriviaQA (train split)...")
    # This confirms the downgrade to datasets==2.19.0 is working for loading scripts
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    print(f"   Success! Loaded {len(ds)} samples.")
except Exception as e:
    print(f"   FAILED: {e}")
    
print("-------------------------------")