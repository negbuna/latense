import argparse
import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from data import get_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

def cluster_dataset(args):
    # load tokenizer to match data loading logic
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # load dataset (usually train split for clustering)
    load_split = "train" if args.split == "dev" else args.split
    dataset = get_dataset(args.dataset, tokenizer, args.prompt_idx, split=load_split)
    
    # create train/dev splits if requested
    if args.split == "train":
        dataset = dataset.select(range(int(len(dataset) * 0.9)))
        print(f"Selected first 90% for training: {len(dataset)} samples")
    elif args.split == "dev":
        dataset = dataset.select(range(int(len(dataset) * 0.9), len(dataset)))
        print(f"Selected last 10% for dev: {len(dataset)} samples")
        
    # limit number of samples if requested
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
        print(f"Limited dataset to {args.max_samples} samples for clustering")

    print(f"loaded {len(dataset)} examples from {args.dataset}")
    
    # embed data
    print(f"loading embedder: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model, device=args.device)
    texts = dataset["formatted"]
    
    print("embedding data...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # perform clustering
    print(f"clustering into {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed)
    kmeans.fit(embeddings)
    
    # save centroids
    os.makedirs(args.output_dir, exist_ok=True)
    centroids = torch.tensor(kmeans.cluster_centers_)
    torch.save(centroids, os.path.join(args.output_dir, "centroids.pt"))
    print(f"saved centroids to {os.path.join(args.output_dir, 'centroids.pt')}")
    
    # save indices for each cluster so we can compute vectors later
    labels = kmeans.labels_
    for i in range(args.n_clusters):
        indices = np.where(labels == i)[0]
        
        save_path = os.path.join(args.output_dir, f"cluster_{i}_indices.txt")
        with open(save_path, "w") as f:
            for idx in indices:
                f.write(f"{idx}\n")
        print(f"cluster {i}: {len(indices)} examples -> {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--prompt_idx", type=int, default=0)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit the number of samples used for clustering")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    cluster_dataset(args)