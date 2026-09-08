import os
import torch
import glob
from tabulate import tabulate

def check_progress():
    results_dir = "./results"
    # Configs from the shell script
    datasets = ["openai/gsm8k", "wics/strategy-qa", "mandarjoshi/trivia_qa"]
    models = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it"]
    
    table = []
    
    for model in models:
        model_short = model.split("/")[-1]
        for dataset in datasets:
            data_short = dataset.replace("/", "_")
            
            # Find the config directory
            # Pattern: results/dataset_model_thresh*
            search_pattern = f"{results_dir}/{data_short}_{model_short}_thresh*"
            config_dirs = glob.glob(search_pattern)
            
            if not config_dirs:
                table.append([model_short, data_short, "MISSING", "MISSING", "MISSING"])
                continue
                
            config_dir = config_dirs[0]
            base_path = f"{config_dir}/{model_short}-{dataset.replace('/', '-')}"
            
            # Helper to check status
            def get_status(subpath):
                # Try specific prompt 0 path
                p0_path = f"{base_path}/{subpath}/prompt0/logistics.pt"
                
                if os.path.exists(p0_path):
                    try:
                        data = torch.load(p0_path)
                        if 'results_history' in data:
                            acc = sum(data['results_history']) / len(data['results_history'])
                            return f"Done ({len(data['results_history'])}, {acc:.2%})"
                        elif 'total' in data: # For state collection
                            return f"Done ({data['total']})"
                    except:
                        return "Corrupt"
                return "Pending"

            cot_status = get_status("greedy_cot_eval")
            sc_status = get_status("self_consistency_eval")
            
            # Vector status (State Collection)
            # State collection path is complex: state_collection/layer.../hidden_states/..
            # We look for logistics.pt in any subdir of state_collection
            state_glob = f"{base_path}/state_collection/*/logistics.pt"
            state_files = glob.glob(state_glob)
            vec_status = "Pending"
            if state_files:
                try:
                    data = torch.load(state_files[0])
                    vec_status = f"Done ({data['total']})"
                except:
                    vec_status = "Corrupt"
            
            table.append([model_short, data_short, cot_status, sc_status, vec_status])

    print(tabulate(table, headers=["Model", "Dataset", "CoT (Acc)", "SC (Acc)", "Vectors (Count)"], tablefmt="grid"))

if __name__ == "__main__":
    check_progress()