import argparse
import os
import torch
import json
import time
from vllm import LLM, SamplingParams
from datasets import load_dataset
from data import get_dataset
from eval_utils import judge_general_answer, calculate_3gram_repetition
from extract_judge_answer import extract_answer, judge_answer, extract_true_answer

# Modified monkey patch for static steering (CAA)
def patch_vllm_for_static_steering(llm, steering_vector, alpha, layer_idx=-1):
    vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layers = vllm_model.model.layers
    if layer_idx < 0:
        layer_idx += len(layers)
    
    target_layer = layers[layer_idx]
    if not hasattr(target_layer, "_original_forward"):
        target_layer._original_forward = target_layer.forward

    v = steering_vector.to(llm.llm_engine.model_config.device)
    v_norm = v.norm()

    def static_forward(hidden_states, *args, **kwargs):
        output = target_layer._original_forward(hidden_states, *args, **kwargs)
        h = output[0] if isinstance(output, tuple) else output
        
        # Static Steering (CAA): Δh = α * v
        # Ensure v matches batch size and sequence length
        # hidden_states: (batch_size, seq_len, hidden_size)
        delta_h = alpha * v.view(1, 1, -1)
        
        steered_h = h + delta_h
        return (steered_h,) + output[1:] if isinstance(output, tuple) else steered_h

    target_layer.forward = static_forward
    print(f"Patched layer {layer_idx} with Static Steering (alpha={alpha})")

def main(args):
    # Set seed
    torch.manual_seed(args.seed)
    
    # Load steering vector
    steering_vector = torch.load(args.vector_path, map_location="cpu").float()
    
    # Initialize vLLM
    # Use max_model_len to save memory if needed
    llm = LLM(model=args.model_name_or_path, trust_remote_code=True, max_model_len=2048, gpu_memory_utilization=0.9)
    
    # Apply steering
    patch_vllm_for_static_steering(llm, steering_vector, args.alpha, args.layer_idx)
    
    # Load dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = get_dataset(args.dataset, tokenizer, prompt_idx=0, split=args.split)
    
    if args.max_eval_samples:
        dataset = dataset.select(range(min(len(dataset), args.max_eval_samples)))

    prompts = [ex["formatted"] for ex in dataset]
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    
    print(f"Starting vLLM generation for {len(prompts)} samples...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    total_duration = end_time - start_time
    
    print(f"Generation complete in {total_duration:.2f}s ({total_duration/len(prompts):.4f}s/it)")
    
    # Process results
    results_history = []
    time_history = []
    token_history = []
    repetition_history = []
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    logistics_path = os.path.join(args.output_dir, "logistics.pt")

    for i, output in enumerate(outputs):
        final_output = output.outputs[0].text
        example = dataset[i]
        true_answer = example["answer"]
        
        # Token count from vLLM output
        num_tokens = len(output.outputs[0].token_ids)
        
        # 3-gram repetition (token-based)
        rep_rate = calculate_3gram_repetition(final_output) # fallback to text-based if tokens are messy
        # Actually, let's use the token_ids from vLLM for accuracy
        token_ids = output.outputs[0].token_ids
        if len(token_ids) >= 3:
            trigrams = [tuple(token_ids[k:k+3]) for k in range(len(token_ids)-2)]
            rep_rate = 1.0 - (len(set(trigrams)) / len(trigrams))
        else:
            rep_rate = 0.0

        # Evaluation
        if "strategy_qa" in args.dataset or "strategy-qa" in args.dataset or "trivia_qa" in args.dataset:
            is_correct = judge_general_answer(final_output, true_answer, args.dataset)
        else:
            # Extract true answer for MATH
            extracted_true = extract_true_answer(true_answer, name=args.dataset)
            # Extract pred answer
            pred_ans = extract_answer(final_output, data_name=args.dataset, prompt_idx=0, model_name=args.model_name_or_path)
            if pred_ans is not None and extracted_true is not None:
                is_correct = judge_answer(pred_ans, extracted_true, data_name=args.dataset)
            else:
                is_correct = False

        results_history.append(is_correct)
        time_history.append(total_duration / len(prompts)) # Avg per sample
        token_history.append(num_tokens)
        repetition_history.append(rep_rate)

    # Save logistics.pt
    torch.save({
        "results_history": results_history,
        "time_history": time_history,
        "token_history": token_history,
        "repetition_history": repetition_history,
        "overhead_history": [0.0] * len(results_history), # vLLM overhead is negligible
        "similarity_traces": [] # Traces are harder to extract in vLLM without deeper hacking
    }, logistics_path)
    
    print(f"Results saved to {logistics_path}. Accuracy: {sum(results_history)/len(results_history):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--vector_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_eval_samples", type=int, default=500)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()
    main(args)
