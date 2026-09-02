import torch
import torch.nn.functional as F
import os
from typing import Dict, Optional

class SteeringRouter:
    """
    Manages dynamic selection of steering vectors based on input semantics.
    Uses a lightweight embedding model to route inputs to the nearest vector cluster.
    """
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2", device="cpu"):
        self.device = device
        self.vectors: Dict[int, torch.Tensor] = {}
        self.centroids: Optional[torch.Tensor] = None
        self.embedder = None

        # use lightweight sentence transformer for routing, decouples routing cost from the main LLM
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model_name, device=device)
        except ImportError:
            print("Warning: sentence-transformers not installed. Dynamic routing will be disabled.")

    def load_clusters(self, cluster_dir: str):
        """
        Loads centroids and corresponding steering vectors from disk.
        Expects:
            - centroids.pt: Tensor of shape [num_clusters, embed_dim]
            - vector_{i}.pt: The steering vector for cluster i
        """
        centroids_path = os.path.join(cluster_dir, "centroids.pt")
        if not os.path.exists(centroids_path):
            print(f"Warning: Centroids file not found at {centroids_path}")
            return
            
        self.centroids = torch.load(centroids_path, map_location=self.device)
        
        # load vectors for each cluster
        for filename in os.listdir(cluster_dir):
            if filename.startswith("vector_") and filename.endswith(".pt"):
                # extract index from vector_0.pt
                try:
                    idx = int(filename.split("_")[1].split(".")[0])
                    self.vectors[idx] = torch.load(os.path.join(cluster_dir, filename), map_location=self.device)
                except (ValueError, IndexError):
                    continue

    def route(self, input_text: str) -> Optional[torch.Tensor]:
        """
        Embeds input text, finds nearest centroid, returns associated steering vector.
        """
        if self.embedder is None or self.centroids is None:
            return None
            
        with torch.no_grad():
            emb = self.embedder.encode(input_text, convert_to_tensor=True, device=self.device)
            
        # cosine similarity: (A . B) / (|A| |B|)
        # normalize embeddings and centroids first
        emb = F.normalize(emb, dim=0)
        centroids_norm = F.normalize(self.centroids, dim=1)
        
        # compute scores
        scores = torch.matmul(centroids_norm, emb)
        best_cluster_idx = torch.argmax(scores).item()
        
        return self.vectors.get(best_cluster_idx)
