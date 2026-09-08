import os
import torch
from .steering_controller import SteeringController

class LaTenseGovernor:
    """
    LaTenseGovernor intercepts the transformer residual stream representations
    at a target layer and dynamically modulates the activations using the LaTense equation.
    """
    def __init__(
        self,
        model,
        vector_path: str = None,
        vector: torch.Tensor = None,
        layer: int = -1,
        alpha: float = 0.3,
        similarity_threshold: float = None,
        ablation_mode: str = "full",
        device: str = None
    ):
        self.model = model
        self.layer = layer
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold
        self.ablation_mode = ablation_mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load steering vector
        if vector is not None:
            self.vector = vector.to(self.device)
        elif vector_path is not None:
            resolved_path = vector_path
            # If not a direct local path, check package relative paths
            if not os.path.exists(resolved_path):
                base_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Check different candidate structures
                candidates = [
                    os.path.join(base_dir, "assets", "vectors", os.path.basename(vector_path)),
                    os.path.join(base_dir, "assets", "vectors", f"{os.path.basename(vector_path)}.pt"),
                    os.path.join(base_dir, "..", vector_path),
                ]
                
                for candidate in candidates:
                    if os.path.exists(candidate):
                        resolved_path = candidate
                        break
            
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(
                    f"Steering vector not found at {vector_path} (resolved: {resolved_path}). "
                    f"Please verify the file path or name."
                )
            
            print(f"Loading steering vector from: {resolved_path}")
            self.vector = torch.load(resolved_path, map_location=self.device)
        else:
            raise ValueError("Must provide either 'vector' (torch.Tensor) or 'vector_path' (str).")

    def steering_context(self):
        """
        Returns a context manager that dynamically hooks into the model layers
        and applies the LaTense steering perturbation during forward passes.
        """
        return SteeringController(
            model=self.model,
            steering_vector=self.vector,
            layer_idx=self.layer,
            alpha=self.alpha,
            similarity_threshold=self.similarity_threshold,
            device=self.device,
            ablation_mode=self.ablation_mode
        )
