from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
from vllm_monkey_patch import patch_vllm_for_latense, unpatch_vllm_for_latense, store_original_forward

def run_greedy_cot_baseline_vllm(llm, dataset, n_samples=5):
    """
    Runs a Greedy CoT baseline on a few samples from the dataset using vLLM.
    """
    prompts = []
    for i in range(n_samples):
        prompt = dataset[i]["problem"]
        messages = [{"role": "user", "content": f"Solve the following math problem step-by-step.\nQuestion: {prompt}\nAnswer:"}]
        # vLLM expects a single string as input
        prompts.append(messages[0]["content"])

    sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = dataset[i]["problem"]
        response = output.outputs[0].text
        
        print(f"--- Sample {i+1} ---")
        print(f"Problem: {prompt}")
        print(f"Model Response: {response}")
        print("--------------------")


if __name__ == "__main__":
    # Load the MATH-500 dataset
    math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Load the model using vLLM
    # NOTE: Replace "your-model-name" with the actual model you want to use
    model_name = "meta-llama/Llama-3.1-8B-Instruct" 
    llm = LLM(model=model_name)

    # --- LaTense Monkey-Patching ---
    # 1. Generate a dummy steering vector
    hidden_size = llm.llm_engine.model_config.get_hidden_size()
    steering_vector = torch.randn(hidden_size)
    alpha = 0.3

    # 2. Store the original forward pass
    store_original_forward(llm)

    # 3. Patch the model
    patch_vllm_for_latense(llm, steering_vector, alpha)

    # Run the baseline
    print("\n--- Running Baseline with LaTense Steering ---")
    run_greedy_cot_baseline_vllm(llm, math_dataset)

    # 4. Unpatch the model
    unpatch_vllm_for_latense(llm)
    
    # Run the baseline again without steering to verify unpatching
    print("\n--- Running Baseline without Steering (verifying unpatch) ---")
    run_greedy_cot_baseline_vllm(llm, math_dataset)
