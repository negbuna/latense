import argparse
import json

def load_results(filepath):
    """Loads a JSONL result file into a dictionary keyed by sample_idx."""
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Assuming a single layer is in the file for steered results
            if 'sample_idx' in data:
                results[data['sample_idx']] = data
    return results

def main(args):
    """
    Analyzes and compares two result files to find cases where the steered
    model fails while the baseline succeeds.
    """
    print(f"Loading baseline results from: {args.baseline_file}")
    baseline_results = load_results(args.baseline_file)
    
    print(f"Loading steered results from: {args.steered_file}")
    steered_results = load_results(args.steered_file)

    if not baseline_results:
        print("Baseline results are empty. Exiting.")
        return
    if not steered_results:
        print("Steered results are empty. Exiting.")
        return

    print("
--- Qualitative Failure Analysis ---")
    print("Finding cases where Baseline was CORRECT and LaTense was INCORRECT...")
    
    failure_count = 0
    max_idx = max(max(baseline_results.keys()), max(steered_results.keys()))

    for i in range(max_idx + 1):
        baseline_sample = baseline_results.get(i)
        steered_sample = steered_results.get(i)

        if not baseline_sample or not steered_sample:
            continue

        baseline_correct = baseline_sample.get('is_correct', False)
        steered_correct = steered_sample.get('is_correct', False)

        if baseline_correct and not steered_correct:
            failure_count += 1
            print(f"
{'='*20} FAILURE CASE #{failure_count} (Sample Index: {i}) {'='*20}")
            
            prompt = baseline_sample.get('prompt', 'N/A')
            true_answer = baseline_sample.get('true_answer', 'N/A')
            
            print(f"
[PROMPT]
{prompt}")
            print(f"
[TRUE ANSWER]
{true_answer}")
            
            print("-" * 60)
            
            baseline_gen = baseline_sample.get('generated_text', 'N/A')
            print(f"
[BASELINE GENERATION - CORRECT]
{baseline_gen}")
            
            print("-" * 60)

            steered_gen = steered_sample.get('generated_text', 'N/A')
            print(f"
[LATENSE GENERATION - INCORRECT]
{steered_gen}")
            
            print(f"{'='*60}
")

    if failure_count == 0:
        print("
No failure cases found where the baseline succeeded and LaTense failed.")
    else:
        print(f"
Found {failure_count} failure cases.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qualitative Failure Analysis for LaTense")
    parser.add_argument("--baseline_file", type=str, required=True, help="Path to the JSONL file for the baseline run (e.g., greedy decoding).")
    parser.add_argument("--steered_file", type=str, required=True, help="Path to the JSONL file for the LaTense steered run.")
    args = parser.parse_args()
    main(args)
