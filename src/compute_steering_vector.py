import torch
import os
import argparse
from tqdm import tqdm

def compute_ema(vectors, beta):
    """Computes the Exponential Moving Average of a list of vectors."""
    if not vectors:
        return None
    ema = vectors[0]
    for i in range(1, len(vectors)):
        ema = beta * ema + (1 - beta) * vectors[i]
    return ema

def compute_steering_vector(args):
    good_vectors = []
    bad_vectors = []

    print(f"Scanning directory: {args.states_dir}")
    for filename in tqdm(os.listdir(args.states_dir)):
        if filename.endswith(".pt"):
            file_path = os.path.join(args.states_dir, filename)
            try:
                tensor = torch.load(file_path)
                if "good" in filename:
                    good_vectors.append(tensor)
                elif "bad" in filename:
                    bad_vectors.append(tensor)
            except Exception as e:
                print(f"Could not load or process file {filename}: {e}")

    if not good_vectors or not bad_vectors:
        raise ValueError("Could not find enough good/bad vectors to compute. Please run collect_states.py first.")

    print(f"Found {len(good_vectors)} good vectors and {len(bad_vectors)} bad vectors.")

    if args.averaging_method == 'mean':
        print("Using simple mean averaging.")
        avg_good = torch.stack(good_vectors).mean(dim=0)
        avg_bad = torch.stack(bad_vectors).mean(dim=0)
    elif args.averaging_method == 'ema':
        print(f"Using Exponential Moving Average (EMA) with beta={args.ema_beta}.")
        avg_good = compute_ema(good_vectors, args.ema_beta)
        avg_bad = compute_ema(bad_vectors, args.ema_beta)
    else:
        raise ValueError(f"Unknown averaging method: {args.averaging_method}")

    # compute steering vector
    steering_vector = avg_good - avg_bad

    # create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch.save(steering_vector, args.output_file)
    print(f"Steering vector computed and saved to {args.output_file}")
    print(f"Vector dimension: {steering_vector.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the steering vector from collected hidden states.")
    parser.add_argument("--states_dir", type=str, required=True, help="Directory containing the saved hidden states (.pt files).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final steering vector.")
    parser.add_argument("--averaging_method", type=str, default="mean", choices=["mean", "ema"], help="Averaging method to use.")
    parser.add_argument("--ema_beta", type=float, default=0.9, help="Beta value for EMA smoothing.")
    args = parser.parse_args()

    compute_steering_vector(args)