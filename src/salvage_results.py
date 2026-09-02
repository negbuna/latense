import torch
import glob
import os

def main():
    print("--- Salvaging Valid Baselines (Truncating to 500 samples) ---")
    results_dir = "./results"
    
    # Find all logistics.pt files
    files = glob.glob(f"{results_dir}/**/logistics.pt", recursive=True)
    
    for f in sorted(files):
        # Skip vector collection files
        if "state_collection" in f:
            continue
            
        # Skip StrategyQA (We know these are invalid)
        if "strategy" in f.lower():
            continue
            
        try:
            data = torch.load(f)
            results = data.get("results_history", [])
            
            if not results:
                continue
                
            # Truncate to 500
            target = 500
            if len(results) < target:
                print(f"Warning: {f} only has {len(results)} samples (Target: {target})")
                subset = results
            else:
                subset = results[:target]
            
            acc = sum(subset) / len(subset)
            
            # Clean up path for display
            # ./results/dataset_model_thresh/model-dataset/mode_eval/prompt0/logistics.pt
            parts = f.split("/")
            print(f"[{parts[2]}] {parts[4]}: {acc:.4f} (on {len(subset)} samples)")
            
        except Exception as e:
            print(f"Error reading {f}: {e}")

if __name__ == "__main__":
    main()