import torch
import numpy as np
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import os

def main():
    parser = argparse.ArgumentParser(description="LaTense Deployment Pre-Screening Utility")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model ID")
    parser.add_argument("--dataset", type=str, default="wics/strategy-qa", help="Validation dataset")
    parser.add_argument("--vector_path", type=str, required=True, help="Path to steering vector .pt file")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of screening samples")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"=== Running Pre-Screening for Model: {args.model} ===")
    
    # 1. Load Steering Vector
    if not os.path.exists(args.vector_path):
        print(f"Error: Vector path {args.vector_path} does not exist.")
        return
    vector = torch.load(args.vector_path, map_location="cpu").to(args.device)
    v_norm = vector.norm().item()
    print(f"Loaded Steering Vector. Norm: {v_norm:.4f}")
    
    # 2. Load Model and Tokenizer (with SDPA fallback)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=args.device
        )
    except Exception as e:
        print(f"FlashAttention-2 load failed: {e}. Falling back to sdpa...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map=args.device
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # 3. Load Validation Split
    try:
        ds = load_dataset(args.dataset, split=f"test[:{args.num_samples}]")
    except Exception as e:
        # Fallback to train or mock data
        ds = [{"question": "Is the earth round?"} for _ in range(args.num_samples)]

    # 4. Extract hidden states and compute alignment variance
    num_layers = model.config.num_hidden_layers
    target_layer = num_layers // 2
    print(f"Pre-screening on target layer: {target_layer} / {num_layers}")
    
    layer_hidden_states = []
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        layer_hidden_states.append(h.detach())
        return output

    # Register hook on target layer
    layers = model.model.layers
    hook = layers[target_layer].register_forward_hook(hook_fn)

    print("Running forward passes on validation split...")
    cosines = []
    for idx, item in enumerate(ds):
        q = item.get("question", item.get("input", "Is the earth round?"))
        inputs = tokenizer(q, return_tensors="pt").to(args.device)
        layer_hidden_states.clear()
        
        with torch.no_grad():
            _ = model(**inputs)
        
        if layer_hidden_states:
            h = layer_hidden_states[0] # (1, seq_len, hidden_dim)
            h_flat = h.squeeze(0).to(torch.float32) # (seq_len, hidden_dim)
            v_flat = vector.to(torch.float32)
            
            dot_products = torch.matmul(h_flat, v_flat)
            h_norms = torch.norm(h_flat, dim=-1)
            v_norm_val = torch.norm(v_flat)
            
            cos = dot_products / (h_norms * v_norm_val + 1e-8)
            cosines.extend(cos.cpu().numpy().tolist())

    hook.remove()
    
    # 5. Compute Alignment Variance
    cosines = np.array(cosines)
    mean_cos = np.mean(cosines)
    var_cos = np.var(cosines)
    std_cos = np.std(cosines)
    
    print("\n=== PRE-SCREENING RESULTS ===")
    print(f"Total tokens evaluated: {len(cosines)}")
    print(f"Mean Cosine Alignment: {mean_cos:.4f}")
    print(f"Alignment Standard Deviation: {std_cos:.4f}")
    print(f"Alignment Variance: {var_cos:.6f}")
    
    # 6. Classification
    if var_cos < 0.005:
        print("RISK CLASSIFICATION: HIGH RISK (Low Variance / Packed Manifold)")
        print("Warning: This architecture exhibits characteristics of high representation density.")
        print("Static steering is highly likely to trigger latent collapse (text loops).")
        print("Recommendation: Use LaTense with active similarity gating (alpha <= 0.1).")
    else:
        print("RISK CLASSIFICATION: LOW RISK (Normal Variance / Sparse Manifold)")
        print("This architecture has sufficient orthogonal subspaces.")
        print("Recommendation: Safe to deploy with standard or dynamic steering.")
        
if __name__ == "__main__":
    main()
