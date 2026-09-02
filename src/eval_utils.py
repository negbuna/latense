import re
import string

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_strategyqa_answer(text):
    """
    Extracts Yes/No from the generated text.
    """
    text = text.lower()
    
    # check for explicit patterns first
    patterns = [
        r"^\s*(yes|no)[,.]?$", # Just Yes/No
        r"^\s*(yes|no)[,.]?\s+", # Yes/No at start
        r"the answer is\s*[:\s]*\**\s*(yes|no)",
        r"answer\s*[:\s]*\**\s*(yes|no)",
        r"therefore\s*[,:]?\s*\**\s*(yes|no)",
        r"thus\s*[,:]?\s*\**\s*(yes|no)",
        r"so\s*[,:]?\s*\**\s*(yes|no)",
        r"\**\s*(yes|no)\s*\**\s*[.!]?$" # Yes/No at end with potential markdown
    ]
    
    for p in patterns:
        match = re.search(p, text)
        if match:
            return match.group(1)
            
    # fallback: check the last few tokens
    clean_text = re.sub(r'[^\w\s]', '', text)
    words = clean_text.split()
    if not words:
        return None
        
    if words[-1] in ["yes", "no"]:
        return words[-1]
        
    return None

def judge_general_answer(prediction, ground_truth, dataset_name):
    """
    Judges correctness for non-math datasets.
    """
    if "strategy_qa" in dataset_name or "strategy-qa" in dataset_name:
        # ground truth is boolean in dataset, convert to string
        gt_str = "yes" if ground_truth else "no"
        pred = extract_strategyqa_answer(prediction)
        if pred is None:
            return False
        return normalize_text(pred) == normalize_text(str(gt_str))
        
    elif "trivia_qa" in dataset_name:
        # TriviaQA ground truth is a dict with aliases
        # check if the prediction contains any of the valid aliases
        aliases = ground_truth.get("aliases", []) if isinstance(ground_truth, dict) else [ground_truth]
        norm_pred = normalize_text(prediction)
        return any(normalize_text(alias) in norm_pred for alias in aliases)
        
    return False

def calculate_3gram_repetition(text):
    """
    Calculates the 3-gram repetition rate to monitor for semantic loops.
    Returns 1.0 - (unique 3-grams / total 3-grams).
    """
    tokens = text.split()
    if len(tokens) < 3:
        return 0.0
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
    unique_trigrams = set(trigrams)
    return 1.0 - (len(unique_trigrams) / len(trigrams))
import json
import os

def log_result(file_path, data):
    """
    Appends a result dictionary to a JSONL file.
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
