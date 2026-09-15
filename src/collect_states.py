import argparse
import os
import random
import numpy as np
import torch
from tqdm import tqdm

from data import get_dataset
from extract_judge_answer import extract_true_answer
from eval_utils import judge_general_answer
from ori_generation import original_generation
from rewards.reward import RewardModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Collect hidden states for steering vector computation.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., 'openai/gsm8k', 'MATH-500')")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--solver_prompt_idx", type=int, default=0, help="Index of the solver prompt to use")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to process")
    parser.add_argument("--end_data_idx", type=int, default=None, help="End index of the data to process (exclusive)")
    parser.add_argument("--good_example_threshold", type=int, default=4, choices=[1, 2, 3, 4], help="Number of verifiers that must pass for an example to be 'good'.")
    parser.add_argument("--indices_file", type=str, default=None, help="Path to a file with a list of indices to process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda', 'cpu')")
    parser.add_argument("--resume", action="store_true", help="Resume from a previous run")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer index to collect states from")
    args = parser.parse_args()
    if os.environ.get("LATENSE_RESUME") == "1":
        args.resume = True
    return args

def main(args):
    """Main function to collect hidden states."""
    if args.seed:
        set_seed(args.seed)

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # only load reward model if we arent doing general QA/we know the dataset supports it.
    reward_model = None
    if "strategy_qa" not in args.dataset and "strategy-qa" not in args.dataset and "trivia_qa" not in args.dataset:
        print("Loading reward model...")
        reward_model = RewardModel(
            model=model,
            tokenizer=tokenizer,
            device=device,
            data_name=args.dataset
        )

    print(f"Loading dataset: {args.dataset}, split: {args.split}")
    
    # handle dev split logic manually since get_dataset might not support it
    load_split = "train" if args.split == "dev" else args.split
    
    dataset = get_dataset(
        args.dataset,
        tokenizer=tokenizer,
        prompt_idx=args.solver_prompt_idx,
        split=load_split,
    )

    # create train/dev splits if needed
    if args.split == "train":
        dataset = dataset.select(range(int(len(dataset) * 0.9)))
        print(f"Selected first 90% for training: {len(dataset)} samples")
    elif args.split == "dev":
        dataset = dataset.select(range(int(len(dataset) * 0.9), len(dataset)))
        print(f"Selected last 10% for dev: {len(dataset)} samples")

    # set up directories and logging
    model_name = args.model_name_or_path.rstrip("/").split("/")[-1]
    data_name = args.dataset.replace("/", "-")
    base_output_dir = f"{args.output_dir}/{model_name}-{data_name}"
    
    if "strategy_qa" in args.dataset or "strategy-qa" in args.dataset or "trivia_qa" in args.dataset:
        dir_suffix = f"layer{args.layer_idx}_prompt{args.solver_prompt_idx}"
    else:
        dir_suffix = f"layer{args.layer_idx}_prompt{args.solver_prompt_idx}_thresh{args.good_example_threshold}"

    output_dir = f"{base_output_dir}/state_collection/{dir_suffix}"
    state_dir = f"{output_dir}/hidden_states/"
    os.makedirs(state_dir, exist_ok=True)

    logistics_path = f"{output_dir}/logistics.pt"
    good_count = 0
    bad_count = 0
    start_idx = args.start_data_idx

    if args.resume and os.path.exists(logistics_path):
        print(f"Resuming from {output_dir}")
        logistics = torch.load(logistics_path)
        good_count = logistics.get("good_count", 0)
        bad_count = logistics.get("bad_count", 0)
        start_idx = logistics.get("start_idx", 0)

    # determine data range
    if args.indices_file:
        print(f"loading indices from {args.indices_file}")
        with open(args.indices_file, 'r') as f:
            data_idx_list = [int(line.strip()) for line in f]
        print(f"processing {len(data_idx_list)} samples from indices file.")
    else:
        end_idx = args.end_data_idx if args.end_data_idx is not None else len(dataset)
        end_idx = min(end_idx, len(dataset))
        data_idx_list = range(start_idx, end_idx)
        print(f"starting state collection from index {start_idx} to {end_idx}...")

    for i in tqdm(data_idx_list, desc="Collecting States"):
        example = dataset[i]
        
        if "strategy_qa" in args.dataset or "strategy-qa" in args.dataset or "trivia_qa" in args.dataset:
            true_answer = example["answer"]
        else:
            true_answer = extract_true_answer(example["answer"], name=args.dataset)

        if true_answer is None:
            continue

        # generate one solution
        original_output, hidden_states_list, _ = original_generation(
            input_text=example["formatted"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            layer_idx=args.layer_idx
        )

        if not hidden_states_list:
            continue

        # judge solution
        if "strategy_qa" in args.dataset or "strategy-qa" in args.dataset or "trivia_qa" in args.dataset:
            # For general datasets, simple correctness check
            is_good = judge_general_answer(original_output, true_answer, args.dataset)
        else:
            verifications = reward_model.get_verifications(example["question"], original_output)
            num_verifiers_passed = sum(verifications.values())
            is_good = num_verifiers_passed >= args.good_example_threshold

        # save the averaged hidden state
        status = "good" if is_good else "bad"
        file_path = f"{state_dir}/idx_{i}_{status}.pt"
        avg_hidden_state = torch.stack(hidden_states_list).mean(dim=0)
        torch.save(avg_hidden_state, file_path)

        if is_good:
            good_count += 1
        else:
            bad_count += 1

        # save logistics
        torch.save({
            "good_count": good_count,
            "bad_count": bad_count,
            "total": good_count + bad_count,
            "start_idx": i + 1,
            "args": vars(args),
        }, logistics_path)

    total_processed = good_count + bad_count
    if total_processed > 0:
        print("\nFinished collecting states.")
        print(f"Good examples: {good_count} ({good_count/total_processed:.2%})")
        print(f"Bad examples: {bad_count} ({bad_count/total_processed:.2%})")
    else:
        print("\nNo states were collected.")

if __name__ == "__main__":
    args = parse_arguments()
    print("--- Command Line Arguments ---")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("----------------------------")
    main(args)