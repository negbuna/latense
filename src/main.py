from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from data import get_dataset
from tqdm import tqdm
from rewards.reward import RewardModel
from ori_generation import original_generation
from opt_generation import optimized_generation
from steered_generation import steered_generation
from gated_generation import gated_steered_generation
from baselines import greedy_cot_generation, self_consistency_generation
from steering_controller import SteeringController
from router import SteeringRouter
from contextlib import nullcontext
import os
from extract_judge_answer import extract_answer, extract_true_answer, judge_answer
from eval_utils import judge_general_answer
import argparse
import numpy as np
import random
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., 'openai/gsm8k', 'MATH-500')")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--prompts_to_run", type=str, default="all", help="Which prompts to run, e.g., '0', '1', or 'all'")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=None, help="End index of the data to evaluate (exclusive)")
    parser.add_argument("--fixed_subset_path", type=str, default=None, help="Path to a file with a fixed list of indices for evaluation.")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose print statements")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Maximum number of samples to evaluate (truncates dataset before sharding)")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--do_sample", action="store_true", help="Use sampling during generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")

    # generation mode
    parser.add_argument("--generation_mode", type=str, default="greedy_cot", choices=["latentseek", "latense", "greedy_cot", "self_consistency", "latense_gated"], help="Generation mode to use")

    # latentseek args
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping threshold")
    parser.add_argument("--k", type=float, default=0.1, help="Ratio of update length to the total length of hidden states")
    parser.add_argument("--max_num_steps", type=int, default=10, help="Number of optimization iterations")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Number of generated tokens")
    parser.add_argument("--reward_threshold", type=float, default=-0.2, help="Threshold for reward to stop optimization")

    # steered generation args
    parser.add_argument("--vector_name_template", type=str, default="./vectors/{model_name}_{dataset_name}_p{prompt_idx}.pt", help="Template for steering vector filenames.")
    parser.add_argument("--alpha", type=float, default=0.3, help="Strength of the steering intervention")
    parser.add_argument("--similarity_threshold", type=float, default=None, help="Cosine similarity threshold to trigger steering (default: None, relies on modulation)")
    parser.add_argument("--cluster_dir", type=str, default=None, help="Directory containing centroids and vectors for dynamic routing")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index to apply steering at")
    parser.add_argument("--no_dynamic_scaling", action="store_true", help="Disable dynamic norm-based scaling of the steering vector")
    parser.add_argument("--no_similarity_modulation", action="store_true", help="Disable dynamic alpha scaling based on cosine similarity")

    # random baseline
    parser.add_argument("--random_vector", action="store_true", help="Use a random steering vector for control baseline")

    # self-consistency args
    parser.add_argument("--sc_k", type=int, default=5, help="Number of samples for self-consistency")

    # device
    parser.add_argument("--device", type=str, default=None)

    # format reward
    parser.add_argument("--rule_format_string", type=str, default=None, help="the answer format that should follow")

    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    
    # parallelization args
    parser.add_argument("--num_shards", type=int, default=1, help="Number of shards to split the dataset into")
    parser.add_argument("--shard_id", type=int, default=0, help="Index of the current shard (0 to num_shards-1)")
    args = parser.parse_args()
    if os.environ.get("LATENSE_RESUME") == "1":
        args.resume = True
    return args


def set_seed(seed):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


# evaluate function 
def main(args):
    """
    Evaluate model
    """
    if args.rule_format_string == "boxed":
        rule_format_string = r'\\boxed{(.*)}'
    else:
        if args.rule_format_string:
            raise ValueError("Unknown format")
        rule_format_string = None
    
    if args.seed:
        set_seed(args.seed)
    
    # set device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device
        )
    except Exception as e:
        print(f"FlashAttention-2 load failed: {e}. Falling back to sdpa...")
        model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                device_map=device
        )
    model = torch.compile(model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load reward model if in latentseek mode
    reward_model = None
    if args.generation_mode == "latentseek":
        reward_model = RewardModel(
                model=model, 
                tokenizer=tokenizer, 
                device=device,
                data_name=args.dataset,
                rule_format_string=rule_format_string
                )

    # --- main loop for prompts ---
    # default to just prompt 0 if not specified
    if args.prompts_to_run == 'all':
        prompts_to_run = [0]
    else:
        prompts_to_run = [int(p.strip()) for p in args.prompts_to_run.split(',')]

    final_results = {}

    for prompt_idx in prompts_to_run:
        print(f"\n{'='*20} EVALUATING PROMPT {prompt_idx+1} {'='*20}")

        model_name = args.model_name_or_path.rstrip("/").split("/")[-1]
        data_name = args.dataset.replace("/", "-")

        # setup steering strategy (static vs dynamic)
        steering_vector = None
        router = None

        if args.generation_mode in ["latense", "latense_gated"]:
            if args.cluster_dir:
                print(f"Initializing dynamic router from {args.cluster_dir}")
                router = SteeringRouter(device=device)
                router.load_clusters(args.cluster_dir)
            else:
                # static vector logic
                vector_path = args.vector_name_template.format(
                    model_name=model_name, 
                    dataset_name=data_name, 
                    prompt_idx=prompt_idx
                )
                
                if args.random_vector:
                    print(f"Generating RANDOM steering vector for prompt {prompt_idx}")
                    # Try to load real vector to get magnitude, otherwise use unit norm
                    if os.path.exists(vector_path):
                        real_vec = torch.load(vector_path, map_location=device)
                        # Random direction, same magnitude
                        rand_vec = torch.randn_like(real_vec)
                        steering_vector = rand_vec / rand_vec.norm() * real_vec.norm()
                    else:
                        # Fallback if no vector exists
                        hidden_size = model.config.hidden_size
                        rand_vec = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)
                        steering_vector = rand_vec / rand_vec.norm() # Unit norm
                elif not os.path.exists(vector_path):
                    print(f"\nWarning: Could not find steering vector for prompt {prompt_idx} at {vector_path}. Skipping.")
                    continue
                else:
                    steering_vector = torch.load(vector_path, map_location=device)

        # load dataset for the specific prompt
        load_split = "train" if args.split == "dev" else args.split
        
        dataset = get_dataset(args.dataset, 
                              tokenizer=tokenizer,
                              prompt_idx=prompt_idx,
                              split=load_split)
        
        # Create train/dev splits if requested
        if args.split == "train":
            dataset = dataset.select(range(int(len(dataset) * 0.9)))
            print(f"Selected first 90% for training: {len(dataset)} samples")
        elif args.split == "dev":
            dataset = dataset.select(range(int(len(dataset) * 0.9), len(dataset)))
            print(f"Selected last 10% for dev: {len(dataset)} samples")
            
        # limit total evaluation samples if requested (before sharding)
        if args.max_eval_samples is not None and len(dataset) > args.max_eval_samples:
            dataset = dataset.select(range(args.max_eval_samples))
            print(f"Truncated dataset to {len(dataset)} samples for evaluation")

        
        # filter dataset if a fixed subset is provided
        if args.fixed_subset_path:
            if not os.path.exists(args.fixed_subset_path):
                raise FileNotFoundError(f"Fixed subset file not found at {args.fixed_subset_path}")
            with open(args.fixed_subset_path, 'r') as f:
                indices = [int(line.strip()) for line in f]
            dataset = dataset.select(indices)
            print(f"Evaluating on a fixed subset of {len(dataset)} samples from {args.fixed_subset_path}")

        # apply sharding if requested
        if args.num_shards > 1:
            dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_id)
            print(f"Shard {args.shard_id + 1}/{args.num_shards}: Evaluating {len(dataset)} samples")

        # for logging
        results_history = []
        time_history = []
        token_history = []
        repetition_history = []
        overhead_history = []
        similarity_traces = []
        
        base_output_dir = f"{args.output_dir}/{model_name}-{data_name}"
        output_dir = f"{base_output_dir}/{args.generation_mode}_eval/prompt{prompt_idx}"
        
        if args.num_shards > 1:
            output_dir = f"{output_dir}/shard_{args.shard_id}"
            
        os.makedirs(output_dir, exist_ok=True)

        start_data_idx = 0
        
        if args.resume:
            print(f"Resume from {output_dir}")
            logistics_path = f"{output_dir}/logistics.pt"
            if os.path.exists(logistics_path):
                logistics = torch.load(logistics_path)
                results_history = logistics.get("results_history", [])
                time_history = logistics.get("time_history", [])
                token_history = logistics.get("token_history", [])
                repetition_history = logistics.get("repetition_history", [])
                overhead_history = logistics.get("overhead_history", [])
                similarity_traces = logistics.get("similarity_traces", [])
                
                # Truncate if we are reducing sample size from a previous run
                if args.max_eval_samples and len(results_history) > args.max_eval_samples:
                    print(f"Truncating history from {len(results_history)} to {args.max_eval_samples} samples")
                    results_history = results_history[:args.max_eval_samples]
                    time_history = time_history[:args.max_eval_samples]
                    token_history = token_history[:args.max_eval_samples]
                    repetition_history = repetition_history[:args.max_eval_samples]
                    overhead_history = overhead_history[:args.max_eval_samples]
                    similarity_traces = similarity_traces[:args.max_eval_samples]
                    
                start_data_idx = len(results_history)

        # if using a fixed subset, ignore start/end data_idx from args
        if args.fixed_subset_path:
            start_data_idx = 0
            end_data_idx = len(dataset)
        else:
            start_data_idx = max(start_data_idx, args.start_data_idx)
            end_data_idx = args.end_data_idx if args.end_data_idx is not None else len(dataset)
            end_data_idx = min(end_data_idx, len(dataset))
        
        print(f"Start to evaluate {args.dataset} (Prompt {prompt_idx+1}) from {start_data_idx} to {end_data_idx}...")

        data_idx_list = range(start_data_idx, end_data_idx)
        for i in tqdm(data_idx_list, desc=f"Prompt {prompt_idx+1}"):
            example = dataset[i]
            
            # for math datasets, we extract the true answer using the existing logic
            # for general datasets, we pass the raw answer object to our new judge
            if "strategy_qa" in args.dataset or "strategy-qa" in args.dataset or "trivia_qa" in args.dataset:
                true_answer = example["answer"]
            else:
                true_answer = extract_true_answer(example["answer"], name=args.dataset)
                if true_answer is None:
                    continue

            start_time = time.time()

            if args.generation_mode == "latentseek":
                original_output, hidden_states_list, input_ids = original_generation(
                        input_text=example["formatted"],
                        model=model,
                        tokenizer=tokenizer,
                        device=device,)
                
                final_output, _, _, _, _ = optimized_generation(
                        reward_model=reward_model,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        question=example["question"],
                        input_text=example["formatted"],
                        original_answer=original_output,
                        original_hidden_states_list=hidden_states_list, 
                        input_ids=input_ids,
                        max_num_steps=args.max_num_steps,
                        lr=args.lr,
                        grad_clip=args.grad_clip,
                        k=args.k,
                        reward_threshold=args.reward_threshold,
                )
            elif args.generation_mode == "latense":
                # determine vector for this sample
                current_vector = None
                if router:
                    current_vector = router.route(example["formatted"])
                else:
                    current_vector = steering_vector # fallback to static vector
                
                # if a vector is found, steer; otherwise, use standard generation
                if current_vector is not None:
                    ctx = SteeringController(
                        model, current_vector, layer_idx=args.layer_idx, 
                        alpha=args.alpha, similarity_threshold=args.similarity_threshold, device=device,
                        use_dynamic_scaling=not args.no_dynamic_scaling,
                        use_similarity_modulation=not args.no_similarity_modulation
                    )
                else:
                    ctx = nullcontext()

                with ctx:
                    final_output = greedy_cot_generation(
                        model=model,
                        tokenizer=tokenizer,
                        input_text=example["formatted"],
                        max_new_tokens=args.max_new_tokens,
                        device=device, do_sample=args.do_sample, temperature=args.temperature,
                    )
            elif args.generation_mode == "latense_gated":
                final_output = gated_steered_generation(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=example["formatted"],
                    steering_vector=steering_vector,
                    alpha=args.alpha,
                    similarity_threshold=args.similarity_threshold,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
            elif args.generation_mode == "greedy_cot":
                final_output = greedy_cot_generation(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=example["formatted"],
                    max_new_tokens=args.max_new_tokens,
                    device=device, do_sample=args.do_sample, temperature=args.temperature,
                )
            elif args.generation_mode == "self_consistency":
                # Sanitize data_name for extract_answer utils which expects short names
                sc_data_name = args.dataset
                if "trivia_qa" in args.dataset:
                    sc_data_name = "trivia_qa"
                elif "strategy_qa" in args.dataset or "strategy-qa" in args.dataset:
                    sc_data_name = "strategy_qa"

                final_output = self_consistency_generation(
                    model=model,
                    tokenizer=tokenizer,
                    input_text=example["formatted"],
                    k=args.sc_k,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    data_name=sc_data_name,
                    prompt_idx=prompt_idx,
                    model_name=args.model_name_or_path
                )
            
            duration = time.time() - start_time
            
            # count tokens in the final output (excluding special tokens like BOS/EOS if possible)
            token_ids = tokenizer.encode(final_output, add_special_tokens=False)
            num_tokens = len(token_ids)

            # --- Metrics for NeurIPS Deadline ---
            # 1. 3-gram repetition rate (token-based)
            repetition_rate = 0.0
            if num_tokens >= 3:
                trigrams = [tuple(token_ids[k:k+3]) for k in range(num_tokens-2)]
                unique_trigrams = set(trigrams)
                repetition_rate = 1.0 - (len(unique_trigrams) / len(trigrams))
            
            # 2. Average steering overhead (latency) per token in microseconds
            # Note: ctx will be valid if we are in steering mode
            avg_overhead = 0.0
            cos_sim_trace = []
            if isinstance(ctx, SteeringController):
                if hasattr(ctx, "steering_times") and len(ctx.steering_times) > 0:
                    avg_overhead = sum(ctx.steering_times) / len(ctx.steering_times)
                # 3. For the first 10 samples, save full similarity trace
                if len(results_history) < 10:
                    cos_sim_trace = ctx.cos_sims if hasattr(ctx, "cos_sims") else []

            # evaluation routing
            if "strategy_qa" in args.dataset or "strategy-qa" in args.dataset or "trivia_qa" in args.dataset:
                is_correct = judge_general_answer(final_output, true_answer, args.dataset)
            else:
                # existing math evaluation logic
                final_answer = extract_answer(final_output, 
                                                 data_name=args.dataset, 
                                                 prompt_idx=prompt_idx, 
                                                 model_name=args.model_name_or_path)

                if final_answer is not None:
                    is_correct = judge_answer(
                            final_output, true_answer, data_name=args.dataset, prompt_idx=prompt_idx)
                else:
                    is_correct = False

            results_history.append(is_correct)
            time_history.append(duration)
            token_history.append(num_tokens)
            repetition_history.append(repetition_rate)
            overhead_history.append(avg_overhead)
            if len(cos_sim_trace) > 0:
                similarity_traces.append(cos_sim_trace)
            
            torch.save({
                "results_history": results_history,
                "time_history": time_history,
                "token_history": token_history,
                "repetition_history": repetition_history,
                "overhead_history": overhead_history,
                "similarity_traces": similarity_traces,
            }, f"{output_dir}/logistics.pt")

        total_samples = len(results_history)
        if total_samples > 0:
            final_accuracy = sum(results_history) / total_samples
            avg_time = sum(time_history) / total_samples
            avg_tokens = sum(token_history) / total_samples
            
            final_results[prompt_idx] = {
                "accuracy": final_accuracy, 
                "avg_time": avg_time,
                "avg_tokens": avg_tokens
            }
            print(f"Prompt {prompt_idx+1} Final accuracy: {final_accuracy:.4f}")
            print(f"Prompt {prompt_idx+1} Average generation time: {avg_time:.4f} seconds")
            print(f"Prompt {prompt_idx+1} Average token count: {avg_tokens:.2f}")
        else:
            print(f"No samples were evaluated for Prompt {prompt_idx+1}.")

    # --- print final summary table ---
    print(f"\n{'='*20} FINAL SUMMARY ({args.generation_mode}) {'='*20}")
    print(f"| {'Prompt':<10} | {'Accuracy':<10} | {'Avg. Time':<15} | {'Avg. Tokens':<15} | {'TPS':<10} |")
    print(f"|{'-'*12}|{'-'*12}|{'-'*17}|{'-'*17}|{'-'*12}|")
    for prompt_idx, results in final_results.items():
        tps = results['avg_tokens'] / results['avg_time'] if results['avg_time'] > 0 else 0
        print(f"| {prompt_idx+1:<10} | {results['accuracy']:.4f}   | {results['avg_time']:.4f} sec      | {results['avg_tokens']:.2f}            | {tps:.2f}      |")
    print(f"{ '='*80}")

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\n[Performance] Peak GPU Memory Overhead: {max_mem:.2f} GB")


if __name__ == "__main__":
    args = parse_args()
    main(args)
