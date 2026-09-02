from datasets import load_dataset

print("Downloading MATH-500 dataset...")
load_dataset("HuggingFaceH4/MATH-500", split="test")

print("Downloading StrategyQA dataset...")
load_dataset("wics/strategy-qa", split="test")

print("Downloading TriviaQA dataset...")
load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")

print("All datasets downloaded successfully.")
