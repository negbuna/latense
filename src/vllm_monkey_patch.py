import torch
import torch.nn.functional as F

def patch_vllm_for_latense(llm, steering_vector, alpha, layer_idx=-2):
    """
    Applies a steering vector to a given layer of the model using monkey-patching.

    Args:
        llm: The vLLM model instance.
        steering_vector (torch.Tensor): The vector to apply.
        alpha (float): The strength of the steering intervention.
        layer_idx (int): The index of the layer to patch.
    """
    # 1. Get the model from the vLLM engine
    # This is a bit of a hack, but it's the only way to access the model
    # without modifying the vLLM source code.
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model

    # 2. Get the target layer
    target_layer = vllm_model.model.layers[layer_idx]
    original_forward = target_layer.forward

    def new_forward(hidden_states, *args, **kwargs):
        # Call the original forward pass
        output = original_forward(hidden_states, *args, **kwargs)
        
        # The first element of the output tuple is the hidden states
        h = output[0]
        
        # Apply the steering vector
        # Δh = α * (1 - cos(h, v)) * (||h|| / ||v||) * v
        v = steering_vector.to(h.device)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(h, v.unsqueeze(0).unsqueeze(0), dim=-1)
        
        # Norms
        h_norm = torch.norm(h, dim=-1)
        v_norm = torch.norm(v)
        
        # Scaling factor
        scaling_factor = (h_norm / v_norm)
        
        # Delta h
        delta_h = alpha * (1 - cos_sim) * scaling_factor.unsqueeze(-1) * v.unsqueeze(0).unsqueeze(0)
        
        # Apply the steering
        steered_h = h + delta_h
        
        # Return the modified hidden states
        return (steered_h,) + output[1:]

    # 3. Monkey-patch the forward method
    target_layer.forward = new_forward
    print(f"Successfully patched layer {layer_idx} for LaTense steering.")

def unpatch_vllm_for_latense(llm, layer_idx=-2):
    """
    Restores the original forward method of the specified layer.
    """
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    target_layer = vllm_model.model.layers[layer_idx]
    
    # Check if we have a stored original forward method
    if hasattr(target_layer, "_original_forward"):
        target_layer.forward = target_layer._original_forward
        delattr(target_layer, "_original_forward")
        print(f"Successfully unpatched layer {layer_idx}.")
    else:
        print(f"Warning: Layer {layer_idx} does not appear to be patched.")

def store_original_forward(llm, layer_idx=-2):
    """
    Stores the original forward method before patching.
    """
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    target_layer = vllm_model.model.layers[layer_idx]
    
    if not hasattr(target_layer, "_original_forward"):
        target_layer._original_forward = target_layer.forward

if __name__ == '__main__':
    # This is a placeholder for testing the monkey-patching logic.
    # It will not run without a vLLM model.
    pass
