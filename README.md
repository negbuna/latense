# LaTense: Mitigating Latent Collapse in Activation Steering via Geometric Alignment

[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://nathanegbuna.com/latense/latense_paper.pdf)
[![Website](https://img.shields.io/badge/Website-Interactive_Demo-22c55e.svg)](https://nathanegbuna.com/latense)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](LICENSE)

**LaTense** (**Lat**ent S**ense**) is a training-free activation steering framework that dynamically modulates intervention strength during inference based on the local geometric alignment between the model hidden state and a reasoning vector. 

By applying a cosine penalty $(1 - \cos(h, v))$ and norm-proportional scaling $\frac{\|h\|}{\|v\|}$, LaTense acts as a restorative governor that eliminates text looping (reducing repetition rates to **0.00%** across Llama, Gemma, and Qwen) while achieving a **4.6x reduction in token compute** over sampling-based test-time compute baselines.

---

## Key Scientific Findings

1. **Dynamic Geometric Governance**: Unlike static steering ($h^\prime = h + \alpha v$), LaTense scales interventions per token:
   $$\Delta h = \alpha \cdot (1 - \cos(h, v)) \cdot \frac{\|h\|}{\|v\|} \cdot v$$
   This applies restorative pressure when inference trajectories drift away from the reasoning manifold while attenuating intervention on naturally aligned states to preserve factual circuits.

2. **Looping Elimination**: Standard static steering (CAA) triggers severe text-looping collapse (repetition rates up to 38.07% on dense architectures like Qwen-2.5-7B). LaTense stabilizes latent trajectories and brings repetition rates down to **0.00%** across all tested model families.

3. **Inference Efficiency**: Achieves reasoning gains competitive with Self-Consistency ($k=5$) while operating on a single forward pass ($k=1$), slashing token consumption by **4.6x**.

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
    layer=24,
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

The evaluation pipeline enforces strict data disjointness across vector extraction, hyperparameter sweeps, and final evaluation splits.

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
  title={LaTense: Mitigating Latent Collapse in Activation Steering via Geometric Alignment},
  author={Egbuna, Nathan},
  year={2026},
  note={Preprint},
  url={https://nathanegbuna.com/latense}
}
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
