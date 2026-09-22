# LaTense: Measuring the Limits of Geometric Activation Steering

[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://nathanegbuna.com/latense/latense_paper.pdf)
[![Website](https://img.shields.io/badge/Website-Interactive_Demo-22c55e.svg)](https://nathanegbuna.com/latense)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**LaTense** (**Lat**ent S**ense**) is a training-free activation steering framework that dynamically modulates intervention strength during inference based on the local geometric alignment between the model hidden state and a reasoning vector. 

By applying a cosine penalty $(1 - \cos(h, v))$ and norm-proportional scaling $\frac{\|h\|}{\|v\|}$, LaTense modulates how hard the intervention pushes at each token. Its clearest empirical result is on strategic reasoning with Gemma 2 9B IT, where it reaches **74.20%** on StrategyQA, **+8.6 points over static CAA** ($p < 0.0001$, paired bootstrap), at a single-pass cost of 122.7 generated tokens per problem.

> **Correction notice (v2, September 2026).** An audit of the raw run logs found errors in the first release. The claims of a
> 0.00% repetition rate and of 38.07% static-steering collapse on Qwen-2.5-7B have been **withdrawn**: repetition was never logged
> in the reported LaTense runs, and the Qwen run used an out-of-range layer index, so no steering was applied. Reported accuracies
> now come only from complete runs. See the website for the corrected results table.

---

## Key Scientific Findings

1. **Dynamic Geometric Governance**: Unlike static steering ($h^\prime = h + \alpha v$), LaTense scales interventions per token:
   $$\Delta h = \alpha \cdot (1 - \cos(h, v)) \cdot \frac{\|h\|}{\|v\|} \cdot v$$
   This applies restorative pressure when inference trajectories drift away from the reasoning manifold while attenuating intervention on naturally aligned states to preserve factual circuits.

2. **What the Repetition Metric Measures**: Across 19,268 generations from 56 runs spanning six models (0.5B–9.2B), 3-gram repetition is predicted by output length (Spearman ρ = 0.881), not by steering and not by model scale: mean length predicts mean repetition across models (r = +0.967) while parameter count does not (r = −0.288). At the operating point ($\alpha = 0.3$) steering does **not** raise repetition — unsteered MATH-500 scores 35.82% (Llama) and 23.95% (Gemma) against 35.92% and 23.97% steered. A shuffled-token control (5.93% actual vs 0.20% shuffled) shows the metric detects genuine structural recurrence, such as the restated scaffolding of a long derivation, and cannot distinguish it from degenerate looping. Report the excess over that control, or a length-matched reference, rather than a bare rate.

3. **Behavior Under High Intensity**: Steering intensity is bounded. At $\alpha = 1.0$, logged generations reach 80.9% and 94.4% 3-gram repetition, and this collapse affects LaTense as well as static steering. All reported results use $\alpha = 0.3$, below that boundary.

4. **Inference Efficiency**: Runs on a single forward pass ($k=1$), generating 122.7 tokens per problem on Gemma 2 9B IT StrategyQA against 565.9 for Self-Consistency ($k=5$, five sampled chains averaging 113.2 tokens each) — a **4.6x reduction**, confirmed against the run logs.

---

## Quickstart & Installation

### Requirements
* Python 3.10+
* PyTorch 2.2+
* CUDA-compatible GPU (NVIDIA A100 or H100 recommended for full benchmark replication)

```bash
# Clone the repository
git clone https://github.com/negbuna/latense.git
cd latense

# Install the latense package in editable mode
pip install -e .
```

---

## Python API Usage

The `latense` package provides `LaTenseGovernor`, a lightweight context manager that hooks into Hugging Face `transformers` models to perform dynamic steering with zero overhead:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from latense import LaTenseGovernor

model_id = "google/gemma-2-9b-it"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize the governor with layer and steering coefficient alpha
governor = LaTenseGovernor(
    model=model,
    vector_path="vectors/strategy-qa_gemma-2-9b-it_L-1.pt",
    layer=20,
    alpha=0.3
)

inputs = tokenizer("Question: Could the members of The Police perform lawful arrests? Answer:", return_tensors="pt").to("cuda")

# Generate with dynamic geometric steering
with governor.steering_context():
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Experimental Pipeline & Replication

The evaluation pipeline separates vector extraction, hyperparameter sweeps, and final evaluation splits: MATH-500
(`test[400:]`, `test[200:400]`, `test[:200]`), TriviaQA (`train[200:]` for extraction, `validation` for evaluation) and StrategyQA
(`test[1200:]` for extraction, `test[:500]` for evaluation). One exception: the prompt-count sensitivity sweep re-extracts vectors
through the CLI's default split handling, which for StrategyQA falls back to `test` and overlaps that sweep's own evaluation slice.

### 1. Vector Extraction
Extract contrastive reasoning vectors ($v = \mu_{correct} - \mu_{incorrect}$) from holdout training splits:
```bash
python src/create_steering_vectors_cli.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "MATH-500" \
    --task_name "MATH-500" \
    --split "test[400:]" \
    --num_samples 100 \
    --output_dir "./vectors"
```

### 2. Hyperparameter Sweeps
Run layer sensitivity and alpha coefficient sweeps:
```bash
# Layer sensitivity sweep
python src/run_layer_sensitivity.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "MATH-500" \
    --split "test[200:400]" \
    --vector_name "MATH-500" \
    --output_dir "./results"

# Alpha coefficient sweep
python src/run_alpha_sweep.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "MATH-500" \
    --split "test[200:400]" \
    --vector_name "MATH-500" \
    --layer_idx 24 \
    --output_dir "./results"
```

### 3. Evaluation
Evaluate LaTense on the final evaluation test split:
```bash
python src/main.py \
    --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
    --dataset "MATH-500" \
    --split "test[:200]" \
    --generation_mode "latense" \
    --alpha 0.1 \
    --layer_idx 24 \
    --output_dir "./results"
```

---

## Pre-extracted Vectors

Pre-extracted steering vectors for all three model families are bundled in the [`vectors/`](vectors/) directory:
* `vectors/strategy-qa_*.pt` (StrategyQA)
* `vectors/MATH-500_*.pt` (MATH-500)
* `vectors/trivia_qa_*.pt` (TriviaQA)

---

## Citation

If you find LaTense useful in your research, please cite:

```bibtex
@misc{egbuna2026latense,
  title={LaTense: Measuring the Limits of Geometric Activation Steering},
  author={Egbuna, Nathan},
  year={2026},
  note={Preprint},
  url={https://nathanegbuna.com/latense}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
