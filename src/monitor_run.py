import torch
import argparse
import os
import numpy as np

def monitor_run(log_file):
    """
    Loads a logistics.pt file and prints the current progress.
    """
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return

    try:
        log_data = torch.load(log_file)
        results_history = log_data.get("results_history", [])
        time_history = log_data.get("time_history", [])
    except Exception as e:
        print(f"Error loading or parsing log file: {e}")
        return

    samples_completed = len(results_history)

    if samples_completed == 0:
        print("No samples have been completed yet.")
        return

    current_accuracy = np.mean(results_history)
    current_avg_time = np.mean(time_history)

    print("\n--- Live Run Progress ---")
    print(f"Log File: {log_file}")
    print("-------------------------")
    print(f"Samples Completed: {samples_completed}")
    print(f"Current Accuracy:  {current_accuracy:.4f} ({current_accuracy:.2%})")
    print(f"Current Avg. Time: {current_avg_time:.2f} seconds/sample")
    print("-------------------------\
")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor the live progress of an evaluation run.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the logistics.pt file to monitor.")
    args = parser.parse_args()

    monitor_run(args.log_file)
