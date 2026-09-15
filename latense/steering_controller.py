import torch
from torch import nn
import time

class SteeringController:
    """
    A context manager to apply latent steering to a model's activations.
    Supports different ablation modes for systematic evaluation and logs metrics.
    """
    def __init__(self, model, steering_vector, layer_idx=-1, alpha=0.3, similarity_threshold=None, device="cuda", ablation_mode=None, use_dynamic_scaling=True, use_similarity_modulation=True):
        """
        Initializes the SteeringController.
        Args:
            model: The model to be steered.
            steering_vector: The vector to steer the activations with.
            layer_idx: The index of the layer to hook into.
            alpha: The steering strength.
            similarity_threshold: If set, only applies steering if cosine similarity is below this value.
            device: The device to run on.
            ablation_mode (str): Shortcut for setting scaling/modulation.
            use_dynamic_scaling (bool): Whether to scale perturbation by activation norm.
            use_similarity_modulation (bool): Whether to scale alpha by (1 - cos_sim).
        """
        self.model = model
        self.steering_vector = steering_vector.to(device) if steering_vector is not None else None
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # Lists to store metrics for Phase 4 & 5 mechanism analysis
        self.cos_sims = []
        self.delta_h_norms = []
        self.forward_times = []
        self.steering_times = []

        # Set ablation modes or use direct flags
        if ablation_mode == "full":
            self.use_dynamic_scaling = True
            self.use_similarity_modulation = True
        elif ablation_mode == "norm_only":
            self.use_dynamic_scaling = True
            self.use_similarity_modulation = False
        elif ablation_mode == "static":
            self.use_dynamic_scaling = False
            self.use_similarity_modulation = False
        elif ablation_mode is None:
            self.use_dynamic_scaling = use_dynamic_scaling
            self.use_similarity_modulation = use_similarity_modulation
        else:
            raise ValueError(f"Unknown ablation mode: {ablation_mode}")
        
        # pre-calculate normalized vector for cosine similarity
        if self.steering_vector is not None:
            self.vector_norm = self.steering_vector.norm()
            self.normalized_vector = self.steering_vector / (self.vector_norm + 1e-8)

    def __enter__(self):
        if self.steering_vector is None:
            return self
            
        # find the layers module
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            print("Warning: Could not find layers to hook in SteeringController.")
            return self

        # handle negative indexing
        if self.layer_idx < 0:
            self.layer_idx += len(layers)
            
        if 0 <= self.layer_idx < len(layers):
            self.target_layer = layers[self.layer_idx]
            self.original_forward = self.target_layer.forward
            self.target_layer.forward = self._patched_forward
        else:
            print(f"Warning: Layer index {self.layer_idx} out of bounds.")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "original_forward"):
            self.target_layer.forward = self.original_forward
            delattr(self, "original_forward")

    def _patched_forward(self, *args, **kwargs):
        # call the original forward pass and measure time
        start_fwd = time.perf_counter()
        output = self.original_forward(*args, **kwargs)
        end_fwd = time.perf_counter()
        self.forward_times.append((end_fwd - start_fwd) * 1_000_000) # microseconds

        # handle tuple output
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        # Start measuring steering calculation overhead
        start_steer = time.perf_counter()

        # ensure steering vectors are on the same device as the activation
        if self.steering_vector is not None and self.steering_vector.device != h.device:
            self.steering_vector = self.steering_vector.to(h.device)
            if hasattr(self, "normalized_vector"):
                self.normalized_vector = self.normalized_vector.to(h.device)

        # Calculate similarity if needed for threshold OR modulation
        sim = None
        if self.similarity_threshold is not None or self.use_similarity_modulation or True: # Always calculate for logging
            h_norm = h.norm(dim=-1, keepdim=True)
            h_normalized = h / (h_norm + 1e-8)
            # (B, S, D) @ (D,) -> (B, S)
            sim = torch.matmul(h_normalized, self.normalized_vector.view(-1))
            
            # Log the mean cosine similarity across the batch/sequence length for this step
            self.cos_sims.append(sim.mean().item())

        # 1. calculate mask (similarity threshold)
        if self.similarity_threshold is not None:
            # apply steering if similarity < threshold
            mask = (sim < self.similarity_threshold).unsqueeze(-1)
        else:
            mask = 1.0

        # Determine effective alpha
        effective_alpha = self.alpha
        if self.use_similarity_modulation:
            # Scale alpha based on misalignment: (1 - sim)
            # If sim=1 (aligned), factor=0. If sim=0, factor=1.
            # modulation is (B, S, 1) to match (B, S, D)
            modulation = (1.0 - sim).unsqueeze(-1)
            effective_alpha = effective_alpha * modulation

        # 2. calculate perturbation
        if self.use_dynamic_scaling:
            # dynamic: scale alpha relative to current activation norm
            # |perturbation| = alpha * |h|
            current_norms = h.norm(dim=-1, keepdim=True)
            scale = effective_alpha * current_norms / (self.vector_norm + 1e-8)
            perturbation = scale * self.steering_vector.view(1, 1, -1)
        else:
            # if its fixed: |perturbation| = alpha * |v| (effectively constant)
            perturbation = effective_alpha * self.steering_vector.view(1, 1, -1)

        intervention = mask * perturbation
        # Stop measuring steering time
        end_steer = time.perf_counter()
        self.steering_times.append((end_steer - start_steer) * 1_000_000) # microseconds
        
        # Log the mean delta h norm
        self.delta_h_norms.append(intervention.norm(dim=-1).mean().item())

        # return modified output
        if isinstance(output, tuple):
            return (h + intervention,) + output[1:]
        else:
            return h + intervention
