import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from collections import Counter
import time
import math

from extract_judge_answer import extract_answer
from eval_utils import extract_strategyqa_answer, normalize_text

stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

def greedy_cot_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    max_new_tokens: int = 1024,
    device: str = "cuda",
    do_sample: bool = False,
    temperature: float = 0.7,
    return_dict: bool = False
):
    """
    Generates a single, greedy response from the model (standard Chain of Thought).
    Can also be used for sampling if do_sample is True.
    """
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    
    # ensure pad_token_id is set to avoid warnings
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True if return_dict else False,
            output_scores=True if return_dict else False,
        )
    end_time = time.time()

    generation_time = end_time - start_time
    
    if return_dict:
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if outputs.scores and len(outputs.scores) > 0:
            log_probs = []
            for i, logits in enumerate(outputs.scores):
                probs = torch.nn.functional.log_softmax(logits, dim=-1)
                token_id = generated_ids[i]
                log_prob = probs[0, token_id].item()
                log_probs.append(log_prob)
            avg_log_prob = sum(log_probs) / len(log_probs)
            perplexity = math.exp(-avg_log_prob)
        else:
            perplexity = float('inf')
            
        num_tokens = len(generated_ids)
        time_per_token = generation_time / num_tokens if num_tokens > 0 else 0.0
        
        return {
            "text": generated_text,
            "perplexity": perplexity,
            "time_per_token": time_per_token,
            "total_time": generation_time,
            "num_tokens": num_tokens
        }
    else:
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

def self_consistency_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: str,
    k: int = 5,
    max_new_tokens: int = 1024,
    device: str = "cuda",
    data_name: str = "",
    prompt_idx: int = 0,
    model_name: str = ""
) -> str:
    """
    Generates k responses and returns the one with the majority vote answer.
    Optimized for A100 to generate all k reasoning paths in a single batch.
    """
    inputs = tokenizer([input_text], return_tensors="pt").to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generations = []
    # outputs is tensor of shape (k, sequence_length) if return_dict_in_generate=False
    for seq in outputs:
        generated_ids = seq[inputs.input_ids.shape[1]:]
        generations.append(tokenizer.decode(generated_ids, skip_special_tokens=True))

    answers = []
    for gen in generations:
        if "strategy_qa" in data_name or "strategy-qa" in data_name or "strategy_qa" in data_name:
            ans = extract_strategyqa_answer(gen)
        elif "trivia_qa" in data_name:
            ans = normalize_text(gen)
        else:
            ans = extract_answer(
                gen,
                data_name=data_name,
                prompt_idx=prompt_idx,
                model_name=model_name
            )
        answers.append(str(ans) if ans is not None else None)

    if not answers:
        return generations[0] if generations else ""

    majority_vote = Counter(answers).most_common(1)[0][0]

    for i, ans in enumerate(answers):
        if ans == majority_vote:
            return generations[i]
    
    return generations[0]
