import os
import torch
import argparse
from tabulate import tabulate  # optional for pretty printing otherwise format manually

def main(args):
    results = []
    
    print(f"Scanning {args.output_dir} for results...")
    
    for root, dirs, files in os.walk(args.output_dir):
        if "logistics.pt" in files:
            path = os.path.join(root, "logistics.pt")
            try:
                data = torch.load(path)
                
                # check if this is an eval file (has results_history) or collection file
                if "results_history" in data:
                    # eval results
                    acc = sum(data["results_history"]) / len(data["results_history"])
                    avg_time = sum(data["time_history"]) / len(data["time_history"])
                    avg_tokens = sum(data["token_history"]) / len(data["token_history"])
                    
                    # parse path for metadata (e.g., model-dataset/mode/prompt)
                    parts = root.split(os.sep)
                    
                    # Handle sharded paths: .../model-dataset/mode_eval/promptX/shard_Y
                    if parts[-1].startswith("shard_"):
                        prompt = parts[-2]
                        mode = parts[-3].replace("_eval", "")
                        dataset_model = parts[-4]
                    # Standard path: .../model-dataset/mode_eval/promptX
                    elif len(parts) >= 3:
                        prompt = parts[-1]
                        mode = parts[-2].replace("_eval", "")
                        dataset_model = parts[-3]
                    else:
                        dataset_model = "unknown"
                        mode = "unknown"
                        prompt = "unknown"

                    results.append({
                        "Model/Dataset": dataset_model,
                        "Mode": mode,
                        "Prompt": prompt,
                        "Accuracy": f"{acc:.4f}",
                        "Avg Time (s)": f"{avg_time:.2f}",
                        "Avg Tokens": f"{avg_tokens:.1f}",
                        "Samples": len(data["results_history"])
                    })
            except Exception as e:
                print(f"Error reading {path}: {e}")

    # Aggregate shards (group by Model, Dataset, Mode, Prompt)
    if not results:
        print("No results found.")
        return

    grouped = {}
    for r in results:
        key = (r["Model/Dataset"], r["Mode"], r["Prompt"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    final_rows = []
    for key, group in grouped.items():
        total_samples = sum(item["Samples"] for item in group)
        if total_samples == 0: continue

        # Weighted averages
        avg_acc = sum(float(item["Accuracy"]) * item["Samples"] for item in group) / total_samples
        avg_time = sum(float(item["Avg Time (s)"]) * item["Samples"] for item in group) / total_samples
        avg_tokens = sum(float(item["Avg Tokens"]) * item["Samples"] for item in group) / total_samples

        final_rows.append({
            "Model/Dataset": key[0],
            "Mode": key[1],
            "Prompt": key[2],
            "Accuracy": f"{avg_acc:.4f}",
            "Avg Time (s)": f"{avg_time:.2f}",
            "Avg Tokens": f"{avg_tokens:.1f}",
            "Samples": total_samples
        })

    # sort by dataset then mode
    final_rows.sort(key=lambda x: (x["Model/Dataset"], x["Mode"]))

    headers = ["Model/Dataset", "Mode", "Prompt", "Accuracy", "Avg Time (s)", "Avg Tokens", "Samples"]
    rows = [[r[h] for h in headers] for r in final_rows]

    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        # fallback if tabulate is not installed
        print(f"{' | '.join(headers)}")
        for r in rows:
            print(f"{' | '.join(map(str, r))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Root directory of results")
    args = parser.parse_args()
    main(args)