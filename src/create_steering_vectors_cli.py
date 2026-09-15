import torch
import argparse
import os
import pickle
from latense import latense

def main():
    parser = argparse.ArgumentParser(description="Extract Steering Vectors for LaTense")
    parser.add_argument("--model", type=str, required=True, help="HF model name")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset name")
    parser.add_argument("--task_name", type=str, required=True, help="Identifier for the vector")
    parser.add_argument("--num_samples", type=int, default=200, help="Samples to analyze")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--layer_idx", type=int, default=-1, help="Layer to extract (default: last)")
    parser.add_argument("--output_dir", type=str, default="./vectors", help="Where to save")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use the latense engine to create the vector
    engine = latense(args.model)
    
    # This captures activations and calculates the mean(good) - mean(bad)
    # create_vector stores it in engine.vectors internally but doesn't return it
    # We need to save it to disk.
    engine.create_vector(
        task_name=args.task_name,
        dataset_name=args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        layer_idx=args.layer_idx
    )
    
    # Save the computed vector
    vector_path = os.path.join(args.output_dir, f"{args.task_name}_{args.model.split('/')[-1]}_L{args.layer_idx}.pt")
    
    # Access the vector from the engine (it's stored in self.vectors[task_name])
    if args.task_name in engine.vectors:
        torch.save(engine.vectors[args.task_name], vector_path)
        print(f"SUCCESS: Saved vector to {vector_path}")
    else:
        print("ERROR: Vector calculation failed (likely zero correct/incorrect samples).")

if __name__ == "__main__":
    main()
