from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

def run_greedy_cot_baseline(model, tokenizer, dataset, n_samples=5):
    """
    Runs a Greedy CoT baseline on a few samples from the dataset.
    """
    total_time = 0
    total_tokens = 0

    for i in range(n_samples):
        prompt = dataset[i]["problem"]
        messages = [{"role": "user", "content": f"Solve the following math problem step-by-step.\nQuestion: {prompt}\nAnswer:"}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
        
        # Generate text
        start_time = time.time()
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=1024, do_sample=False)
        duration = time.time() - start_time
        
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        num_tokens = len(output_ids[0])

        total_time += duration
        total_tokens += num_tokens
        
        print(f"--- Sample {i+1} ---")
        print(f"Problem: {prompt}")
        print(f"Model Response: {response}")
        print(f"Time per token: {duration / num_tokens:.4f} seconds")
        print("--------------------")

    print(f"\n--- Average Time per Token ---")
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Total tokens: {total_tokens}")
    print(f"Average time per token: {total_time / total_tokens:.4f} seconds")
    print("--------------------")

if __name__ == "__main__":
    # Load the MATH-500 dataset
    math_dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # Load the model and tokenizer
    # NOTE: Replace "your-model-name" with the actual model you want to use
    model_name = "meta-llama/Llama-3.1-8B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # Run the baseline
    run_greedy_cot_baseline(model, tokenizer, math_dataset)
