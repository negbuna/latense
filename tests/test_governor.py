import os
import tempfile
import torch
import torch.nn as nn
from latense import LaTenseGovernor, SteeringController, SteeringRouter, latense

class MockLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        # Identity-like weight initialization
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(hidden_dim))
            self.linear.bias.zero_()

    def forward(self, x, *args, **kwargs):
        # Return hidden states (optionally as tuple like transformers layers)
        return (self.linear(x),)

class MockModel(nn.Module):
    def __init__(self, num_layers=4, hidden_dim=16):
        super().__init__()
        self.layers = nn.ModuleList([MockLayer(hidden_dim) for _ in range(num_layers)])

def test_governor_hooking_and_math():
    print("Running LaTenseGovernor hook and math tests...")
    
    hidden_dim = 16
    device = "cpu"
    mock_model = MockModel(num_layers=4, hidden_dim=hidden_dim).to(device)
    
    # Create a dummy steering vector
    steering_vector = torch.zeros(hidden_dim, dtype=torch.float32)
    steering_vector[0] = 2.0  # Nudge primarily along the 1st dimension
    
    # Save to a temporary file to test vector_path loading
    temp_dir = tempfile.gettempdir()
    vector_path = os.path.join(temp_dir, "test_vector.pt")
    torch.save(steering_vector, vector_path)
    
    # Instantiate the governor
    governor = LaTenseGovernor(
        model=mock_model,
        vector_path=vector_path,
        layer=2,
        alpha=0.5,
        device=device
    )
    
    # Create dummy input: batch_size=1, seq_len=1, hidden_dim=16
    x = torch.ones(1, 1, hidden_dim, dtype=torch.float32)
    
    # Verify original layer state
    original_layer2_forward = mock_model.layers[2].forward
    
    # Run original forward pass
    orig_out, = mock_model.layers[2](x)
    assert torch.allclose(orig_out, x), "Original pass should return identity output"
    
    # Enter the context manager
    with governor.steering_context() as controller:
        # Check layer hook is applied
        assert mock_model.layers[2].forward != original_layer2_forward, "Layer 2 forward should be patched"
        assert hasattr(controller, "original_forward"), "Controller should retain reference to original forward"
        
        # Run steered forward pass
        steered_out, = mock_model.layers[2](x)
        
        # Verify the intervention modified the state
        assert not torch.allclose(steered_out, x), "Steered pass output should be modified"
        
        # Validate that the logs were populated in the controller
        assert len(controller.cos_sims) > 0, "Should record cosine similarities"
        assert len(controller.delta_h_norms) > 0, "Should record intervention norms"
        
        print(f"Original hidden state (first 4 dims): {x[0, 0, :4].tolist()}")
        print(f"Steered hidden state (first 4 dims): {steered_out[0, 0, :4].tolist()}")
        print(f"Cosine similarity: {controller.cos_sims[0]:.4f}")
        print(f"Intervention norm: {controller.delta_h_norms[0]:.4f}")
        
    # Verify exit cleanup restores the layer
    assert mock_model.layers[2].forward == original_layer2_forward, "Layer 2 forward should be restored"
    assert not hasattr(controller, "original_forward"), "Restored controller should not retain hook attributes"
    
    # Run final original forward pass
    final_out, = mock_model.layers[2](x)
    assert torch.allclose(final_out, x), "Post-context pass should return original identity output"
    
    # Clean up temp file
    if os.path.exists(vector_path):
        os.remove(vector_path)
        
    print("SUCCESS: LaTenseGovernor hooking and math tests passed!")

if __name__ == "__main__":
    test_governor_hooking_and_math()
