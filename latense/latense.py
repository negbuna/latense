import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from contextlib import nullcontext

from .steering_controller import SteeringController

class latense:
    """
    the latense framework: task-adaptive latent steering.
    
    provides a clean api for:
    1. creating steering vectors from datasets (create_vector)
    2. performing steered inference (infer)
    """
    def __init__(self, model_name_or_path, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.model_name = model_name_or_path
        print(f"Loading model: {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model.eval()
        
        # dict to store named steering vectors
        self.vectors = {}

    def create_vector(self, task_name, dataset_name, split="train", num_samples=500, prompt_idx=0, layer_idx=-1):
        """
        Creates a steering vector for a specific task by analyzing model activations.
        """
        # Local imports of benchmarking dependencies
        from data import get_dataset
        from ori_generation import original_generation
        from extract_judge_answer import extract_true_answer, extract_answer, judge_answer
        from eval_utils import judge_general_answer

        print(f"Creating vector for task '{task_name}' using {num_samples} samples from {dataset_name}...")
        
        # handle dev split logic manually
        if split == "dev":
            load_split = "train"
        else:
            load_split = split
            
        dataset = get_dataset(dataset_name, self.tokenizer, prompt_idx, split=load_split)

        # create train/dev splits if requested specifically as "train" or "dev"
        if split == "train":
            dataset = dataset.select(range(int(len(dataset) * 0.9)))
            print(f"Selected first 90% for training: {len(dataset)} samples")
        elif split == "dev":
            dataset = dataset.select(range(int(len(dataset) * 0.9), len(dataset)))
            print(f"Selected last 10% for dev: {len(dataset)} samples")
        else:
            print(f"Using custom or provided split: {split}")
        
        good_vectors = []
        bad_vectors = []
        
        # limit to requested samples
        indices = range(min(len(dataset), num_samples))
        
        for i in tqdm(indices, desc=f"Analyzing {task_name}"):
            example = dataset[i]
            
            # determine ground truth
            if "strategy_qa" in dataset_name or "strategy-qa" in dataset_name or "trivia_qa" in dataset_name:
                true_answer = example["answer"]
            else:
                true_answer = extract_true_answer(example["answer"], name=dataset_name)
                
            if true_answer is None:
                continue

            # run generation and capture states
            gen_output, hidden_states_list, _ = original_generation(
                input_text=example["formatted"],
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                layer_idx=layer_idx
            )
            
            if not hidden_states_list:
                continue
                
            # judge correctness
            if "strategy_qa" in dataset_name or "strategy-qa" in dataset_name or "trivia_qa" in dataset_name:
                is_correct = judge_general_answer(gen_output, true_answer, dataset_name)
            else:
                final_ans = extract_answer(gen_output, data_name=dataset_name, prompt_idx=prompt_idx, model_name=self.model_name)
                
                if final_ans is not None:
                    is_correct = judge_answer(gen_output, true_answer, data_name=dataset_name, prompt_idx=prompt_idx)
                else:
                    is_correct = False

            # compute average hidden state for this sample
            avg_state = torch.stack(hidden_states_list).mean(dim=0)
            
            if is_correct:
                good_vectors.append(avg_state)
            else:
                bad_vectors.append(avg_state)
        
        if not good_vectors or not bad_vectors:
            print(f"Warning: Insufficient samples for task '{task_name}'. Good: {len(good_vectors)}, Bad: {len(bad_vectors)}")
            return
            
        mean_good = torch.stack(good_vectors).mean(dim=0)
        mean_bad = torch.stack(bad_vectors).mean(dim=0)
        
        # isolate reasoning direction: (good - bad)
        steering_vector = mean_good - mean_bad
        self.vectors[task_name] = steering_vector.to(self.device)
        print(f"Vector for '{task_name}' created successfully. Norm: {torch.norm(steering_vector):.4f}")

    def infer(self, prompt, steering_task=None, alpha=0.3, max_new_tokens=512, layer_idx=-1, ablation_mode="full"):
        """
        Generates a response, optionally applying latent steering.
        """
        # Local imports of benchmarking dependencies
        from baselines import greedy_cot_generation

        formatted_prompt = prompt

        ctx = nullcontext()
        is_steered = steering_task and steering_task in self.vectors
        if is_steered:
            ctx = SteeringController(
                self.model, 
                self.vectors[steering_task], 
                layer_idx=layer_idx, 
                alpha=alpha, 
                ablation_mode=ablation_mode
            )
            
        with ctx as controller:
            result = greedy_cot_generation(
                model=self.model,
                tokenizer=self.tokenizer,
                input_text=formatted_prompt,
                max_new_tokens=max_new_tokens,
                device=self.device,
                return_dict=True
            )
            
            if is_steered and hasattr(controller, 'cos_sims'):
                cos_sims = controller.cos_sims
                delta_h_norms = controller.delta_h_norms
                fwd_times = controller.forward_times
                steer_times = controller.steering_times
                
                result["avg_cos_sim"] = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
                result["avg_delta_h_norm"] = sum(delta_h_norms) / len(delta_h_norms) if delta_h_norms else 0.0
                result["avg_forward_time_us"] = sum(fwd_times) / len(fwd_times) if fwd_times else 0.0
                result["avg_steering_time_us"] = sum(steer_times) / len(steer_times) if steer_times else 0.0
                
                result["cos_sims"] = cos_sims
                result["delta_h_norms"] = delta_h_norms
            else:
                result["avg_cos_sim"] = 0.0
                result["avg_delta_h_norm"] = 0.0
                result["avg_forward_time_us"] = 0.0
                result["avg_steering_time_us"] = 0.0
                result["cos_sims"] = []
                result["delta_h_norms"] = []
                
            return result
