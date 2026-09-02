import argparse
import csv
import os
import torch
from tqdm import tqdm

from latense import latense
from data import get_dataset
from extract_judge_answer import extract_true_answer, extract_answer, judge_answer
from eval_utils import judge_general_answer, log_result, calculate_3gram_repetition

def main(args):
    """
    Runs a layer sensitivity analysis by applying steering at different layers
    and logging the accuracy for each layer.
    """
    print("Starting layer sensitivity analysis...")
    
    # Initialize latense
    lt = latense(args.model_name_or_path, device=args.device)

    # --- Create or load steering vector ---
    vector_name = args.vector_name
    vector_path = args.vector_path if hasattr(args, 'vector_path') and args.vector_path else f"{vector_name}.pt"
    
    if os.path.exists(vector_path):
        print(f"Loading existing steering vector: {vector_path}")
        lt.vectors[vector_name] = torch.load(vector_path)
    else:
        print(f"Creating steering vector '{vector_name}'...")
        lt.create_vector(
            task_name=vector_name,
            dataset_name=args.vector_dataset,
            split="train",
            num_samples=args.vector_samples,
            layer_idx=-1
        )
        torch.save(lt.vectors[vector_name], vector_path)
        print(f"Saved steering vector to {vector_path}")

    # --- Load test dataset ---
    print(f"Loading test dataset: {args.dataset}")
    # Use prompt_idx 0 for consistency
    dataset = get_dataset(args.dataset, lt.tokenizer, prompt_idx=0, split=args.split)

    # --- Prepare results file ---
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)
    model_name_safe = args.model_name_or_path.replace('/', '_')
    results_file = os.path.join(results_dir, f"layer_sensitivity_{model_name_safe}.jsonl")
    csv_results_file = os.path.join(results_dir, f"layer_sensitivity_{model_name_safe}.csv")
    print(f"Detailed results will be saved to: {results_file}")
    print(f"Summary CSV will be saved to: {csv_results_file}")
    
    if os.path.exists(results_file):
        os.remove(results_file)
        
    with open(csv_results_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["layer_idx", "accuracy", "correct_count", "total_count", "ablation_mode", "avg_ppl", "avg_time_per_token", "avg_cos_sim", "avg_delta_h_norm", "avg_forward_time_us", "avg_steering_time_us", "avg_repetition_rate"])

    # --- Loop over layers and evaluate ---
    if args.layers:
        layer_range = [int(l.strip()) for l in args.layers.split(',')]
    else:
        layer_range = range(args.start_layer, args.end_layer + 1)
        
    print(f"Testing layers: {layer_range}")

    for layer_idx in layer_range:
        print(f"--- Evaluating Layer {layer_idx} ---")
        correct_count = 0
        total_count = 0
        total_ppl = 0.0
        total_time_per_token = 0.0
        total_cos_sim = 0.0
        total_delta_h = 0.0
        total_forward_time_us = 0.0
        total_steering_time_us = 0.0
        total_repetition_rate = 0.0

        num_samples = min(len(dataset), args.num_samples)
        
        for i in tqdm(range(num_samples), desc=f"Layer {layer_idx} Eval"):
            example = dataset[i]
            
            gen_dict = lt.infer(
                prompt=example["formatted"],
                steering_task=vector_name,
                layer_idx=layer_idx,
                alpha=args.alpha,
                ablation_mode=args.ablation_mode,
                max_new_tokens=args.max_new_tokens
            )
            generated_text = gen_dict["text"]
            perplexity = gen_dict["perplexity"]
            time_per_token = gen_dict["time_per_token"]
            avg_cos_sim_token = gen_dict.get("avg_cos_sim", 0.0)
            avg_delta_h_token = gen_dict.get("avg_delta_h_norm", 0.0)
            avg_forward_time_us_token = gen_dict.get("avg_forward_time_us", 0.0)
            avg_steering_time_us_token = gen_dict.get("avg_steering_time_us", 0.0)
            repetition_rate_token = calculate_3gram_repetition(generated_text)

            is_correct = False
            true_answer_text = None
            if "strategy_qa" in args.dataset or "strategy-qa" in args.dataset or "trivia_qa" in args.dataset:
                true_answer = example["answer"]
                true_answer_text = str(true_answer)
                is_correct = judge_general_answer(generated_text, true_answer, args.dataset)
            else:
                true_answer = extract_true_answer(example["answer"], name=args.dataset)
                if true_answer is not None:
                    true_answer_text = str(true_answer)
                    final_ans = extract_answer(generated_text, data_name=args.dataset, model_name=lt.model_name)
                    if final_ans is not None:
                        is_correct = judge_answer(final_ans, true_answer, data_name=args.dataset)
            
            if true_answer is None:
                continue

            log_data = {
                "layer_idx": layer_idx,
                "sample_idx": i,
                "prompt": example["formatted"],
                "generated_text": generated_text,
                "true_answer": true_answer_text,
                "is_correct": is_correct,
                "perplexity": perplexity,
                "time_per_token": time_per_token,
                "avg_cos_sim": avg_cos_sim_token,
                "avg_delta_h_norm": avg_delta_h_token
            }
            log_result(results_file, log_data)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            if perplexity != float('inf'):
                total_ppl += perplexity
            total_time_per_token += time_per_token
            total_cos_sim += avg_cos_sim_token
            total_delta_h += avg_delta_h_token
            total_forward_time_us += avg_forward_time_us_token
            total_steering_time_us += avg_steering_time_us_token
            total_repetition_rate += repetition_rate_token
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        avg_ppl = total_ppl / total_count if total_count > 0 else 0.0
        avg_time = total_time_per_token / total_count if total_count > 0 else 0.0
        avg_cos_sim = total_cos_sim / total_count if total_count > 0 else 0.0
        avg_delta_h = total_delta_h / total_count if total_count > 0 else 0.0
        avg_fwd_time = total_forward_time_us / total_count if total_count > 0 else 0.0
        avg_steer_time = total_steering_time_us / total_count if total_count > 0 else 0.0
        avg_rep_rate = total_repetition_rate / total_count if total_count > 0 else 0.0
        
        print(f"Layer {layer_idx} Accuracy: {accuracy:.4f} ({correct_count}/{total_count}) | PPL: {avg_ppl:.2f} | Time/Token: {avg_time:.4f}s | Rep Rate: {avg_rep_rate:.2%}")
        
        with open(csv_results_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([layer_idx, accuracy, correct_count, total_count, args.ablation_mode, avg_ppl, avg_time, avg_cos_sim, avg_delta_h, avg_fwd_time, avg_steer_time, avg_rep_rate])

    print("Layer sensitivity analysis complete.")
    print(f"Detailed results saved in {results_file}")
    print(f"Summary CSV saved in {csv_results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Layer Sensitivity Analysis for LaTense")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use.")
    parser.add_argument("--dataset", type=str, default="cais/gsm8k", help="Dataset for evaluation.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split for evaluation.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate on.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for generation.")
    
    # Layer settings
    parser.add_argument("--layers", type=str, default="", help="Comma-separated list of layers to test (e.g., '22,25,28,31').")
    parser.add_argument("--start_layer", type=int, default=20, help="Start layer for sensitivity analysis.")
    parser.add_argument("--end_layer", type=int, default=32, help="End layer for sensitivity analysis (inclusive).")
    
    # Steering vector settings
    parser.add_argument("--vector_path", type=str, default="", help="Path to existing steering vector.")
    parser.add_argument("--vector_name", type=str, default="strategy_qa", help="Name for the steering vector.")
    parser.add_argument("--vector_dataset", type=str, default="wics/strategy-qa", help="Dataset to create the steering vector from.")
    parser.add_argument("--vector_samples", type=int, default=500, help="Number of samples to create the vector from.")

    # Steering parameters
    parser.add_argument("--alpha", type=float, default=0.3, help="Steering strength (alpha).")
    parser.add_argument("--ablation_mode", type=str, default="full", choices=["full", "norm_only", "static"], help="Ablation mode for steering.")

    # System settings
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on.")
    
    args = parser.parse_args()
    main(args)
