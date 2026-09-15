"""
Data api
"""
from datasets import load_dataset, load_from_disk
from prompts import gsm8k_prompt, MATH_500_prompt
import re

def get_dataset(data_name_or_path, tokenizer, prompt_idx, split="test"):
    """
    Args:
        data_name_or_path: dataset name or path
        tokenizer: tokenizer
        prompt_idx: which query prompt to use
    Returns:
        dataset: dataset
    """

    ### load dataset ### 
    if "gsm8k" in data_name_or_path:
        try:
            dataset = load_from_disk(data_name_or_path)[split]
        except:
            dataset = load_dataset("openai/gsm8k", "socratic")[split if split in ["train", "test"] else "test"]
        question_col = "question"
        answer_col = "answer"

    elif "MATH-500" in data_name_or_path:
        # Standard MATH-500 handling with slicing support
        m = re.match(r"(.*)\[(.*)\]", split)
        if m:
            base_split, slice_str = m.groups()
            dataset = load_dataset("HuggingFaceH4/MATH-500", split=base_split)
            if ":" in slice_str:
                start_str, end_str = slice_str.split(":")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else len(dataset)
                dataset = dataset.select(range(start, end))
            else:
                dataset = dataset.select([int(slice_str)])
        else:
            dataset = load_dataset("HuggingFaceH4/MATH-500", split="test" if split == "train" else split)
        question_col = "problem"
        answer_col = "answer"

    elif "strategy_qa" in data_name_or_path or "strategy-qa" in data_name_or_path:
        # Handle slicing if present in split string
        m = re.match(r"(.*)\[(.*)\]", split)
        if m:
            base_split, slice_str = m.groups()
            dataset = load_dataset("wics/strategy-qa", split=base_split)
            if ":" in slice_str:
                start_str, end_str = slice_str.split(":")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else len(dataset)
                dataset = dataset.select(range(start, end))
            else:
                dataset = dataset.select([int(slice_str)])
        else:
            try:
                dataset = load_dataset("wics/strategy-qa", split=split)
            except Exception:
                if split == "train":
                    print("Strategy-QA has no train split. Using test split instead.")
                    dataset = load_dataset("wics/strategy-qa", split="test")
                else:
                    print("Cache error detected for StrategyQA. Force redownloading...")
                    dataset = load_dataset("wics/strategy-qa", split=split, download_mode="force_redownload")
        question_col = "question"
        answer_col = "answer"

    elif "trivia_qa" in data_name_or_path:
        # Handle slicing if present in split string
        m = re.match(r"(.*)\[(.*)\]", split)
        if m:
            base_split, slice_str = m.groups()
            dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=base_split)
            if ":" in slice_str:
                start_str, end_str = slice_str.split(":")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else len(dataset)
                dataset = dataset.select(range(start, end))
            else:
                dataset = dataset.select([int(slice_str)])
        else:
            try:
                dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
            except Exception:
                print("Cache error detected for TriviaQA. Force redownloading...")
                dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split, download_mode="force_redownload")
        question_col = "question"
        answer_col = "answer"

    else:
        raise ValueError(f"Unsupported dataset: {data_name_or_path}")

    # preprocess dataset
    def preprocess_function(examples):
        formatted = []
        questions = examples[question_col]
        for q in questions:
            if "gsm8k" in data_name_or_path:
                messages = gsm8k_prompt(q, prompt_idx)
            elif "MATH-500" in data_name_or_path:
                messages = MATH_500_prompt(q, prompt_idx)
            elif "strategy_qa" in data_name_or_path or "strategy-qa" in data_name_or_path:
                messages = [{"role": "user", "content": f"Answer the following yes/no question with reasoning.\nQuestion: {q}\nAnswer:"}]
            elif "trivia_qa" in data_name_or_path:
                messages = [{"role": "user", "content": f"Answer the following question concisely.\nQuestion: {q}\nAnswer:"}]
            else:
                raise ValueError(f"Unsupported dataset: {data_name_or_path}")

            if tokenizer.name_or_path and "gemma" in tokenizer.name_or_path.lower():
                if messages and messages[0]["role"] == "system":
                    system_content = messages[0]["content"]
                    new_messages = [dict(m) for m in messages[1:]]
                    if new_messages and new_messages[0]["role"] == "user":
                        new_messages[0]["content"] = f"{system_content}\n\n{new_messages[0]['content']}"
                    else:
                        new_messages.insert(0, {"role": "user", "content": system_content})
                    messages = new_messages

            formatted.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        return {"formatted": formatted, "question": questions, "answer": examples[answer_col]}

    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
    return dataset
