import os
import random
import argparse
from datasets import load_dataset

def create_fixed_subset(dataset_name, split, n_samples, output_dir):
    """
    Creates a text file containing a fixed, random subset of indices from a dataset.
    """
    print(f"Loading dataset {dataset_name} to create a subset of {n_samples} from the {split} split...")
    
    config_name = None
    if "gsm8k" in dataset_name:
        config_name = "socratic"
    
    actual_dataset_name = dataset_name
    if dataset_name == "MATH-500":
        actual_dataset_name = "HuggingFaceH4/MATH-500"

    dataset = load_dataset(actual_dataset_name, name=config_name, split=split)
    dataset_size = len(dataset)

    if n_samples > dataset_size:
        raise ValueError(f"Requested sample size ({n_samples}) is larger than the dataset size ({dataset_size}).")

    # create a list of all possible indices and shuffle it
    all_indices = list(range(dataset_size))
    random.shuffle(all_indices)

    # take the first n_samples indices
    subset_indices = all_indices[:n_samples]

    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{dataset_name.replace('/', '-')}_{split}_{n_samples}.txt"
    save_path = os.path.join(output_dir, file_name)

    with open(save_path, 'w') as f:
        for index in subset_indices:
            f.write(f"{index}\n")

    print(f"Successfully created subset file at: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create fixed random subsets of indices for evaluation.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the Hugging Face dataset (e.g., 'openai/gsm8k')")
    parser.add_argument("--split", type=str, required=True, help="Dataset split to use (e.g., 'test')")
    parser.add_argument("--n_samples", type=int, required=True, help="The number of samples to include in the subset.")
    parser.add_argument("--output_dir", type=str, default="./subsets", help="Directory to save the subset files.")
    
    # set a seed for reproducibility of the random sample
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling.")

    args = parser.parse_args()
    random.seed(args.seed)

    create_fixed_subset(args.dataset_name, args.split, args.n_samples, args.output_dir)