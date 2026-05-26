# Large Language Models

The `olaverse.llm` module provides clean interfaces for running computationally intensive transformer-based models, including large language models and deep neural language detectors.

---

## LegalPeace — Legal Contract Reasoning (Beta)

!!! warning "Beta Model"
    **LegalPeace is a research/beta model.** It is **not** recommended for production use.
    Always verify all outputs with a qualified legal professional. Trained primarily on U.S. legal data.

`LegalPeace` is a fine-tuned **Mistral-7B-v0.3** model optimized for contract analysis and legal reasoning. It is loaded via `unsloth` for fast, memory-efficient 4-bit quantized inference on GPU.

**Model Card**: [olaverse/legal-peace-v1.0](https://huggingface.co/olaverse/legal-peace-v1.0)

### Model Details

| Property | Value |
|---|---|
| Base Model | Mistral-7B-v0.3 |
| Parameters | 7B |
| Quantization | 4-bit (via unsloth) |
| Training | SFT (4,800 cases) + DPO (419 examples) |
| License | Apache 2.0 |

### Performance vs Base Mistral-7B

| Benchmark | Improvement |
|---|---|
| Inference Speed | ⚡ 10.3% faster |
| Contract Analysis | 📋 32.6% faster |
| Case Predictions | ⚖️ 14.0% faster |

### Installation

```bash
pip install unsloth
# or, for the olaverse[legal] extras:
pip install olaverse[legal]
```

### Usage via Olaverse

```python
from olaverse.llm import LegalPeace

model = LegalPeace()  # defaults to "olaverse/legal-peace-v1.0"
model.load()          # downloads & loads with unsloth (requires GPU + unsloth)

prompt = "Analyze this clause: 'All disputes shall be resolved through binding arbitration in Delaware.' What are the key implications?"
response = model.generate(prompt, max_new_tokens=300, temperature=0.7)
print(response)
```

### Usage via Hugging Face (direct)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="olaverse/legal-peace-v1.0",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

prompt = "Analyze this clause: 'All disputes shall be resolved through binding arbitration in Delaware.'"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Supported Use Cases

- ✅ Contract clause analysis and review
- ✅ Legal research assistance
- ✅ Evidence evaluation
- ✅ Case outcome prediction
- ✅ Legal Q&A

::: olaverse.llm.LegalPeace

---

## LIDNeural5 — Neural Language Identification

`LIDNeural5` is a high-accuracy transformer sequence classifier for identifying 5 major Nigerian languages. It is fine-tuned on top of **`castorini/afriberta_large`** — a multilingual XLM-RoBERTa transformer pre-trained specifically on African languages.

**Model Card**: [olaverse/lid-neural-5](https://huggingface.co/olaverse/lid-neural-5)

### Model Details

| Property | Value |
|---|---|
| Base Model | `castorini/afriberta_large` (XLM-RoBERTa) |
| Parameters | 125M |
| Architecture | Transformer Sequence Classification |
| Model Size | 484 MB |
| License | Apache 2.0 |

### Accuracy

| Language | Precision | Recall | F1-Score |
|---|---|---|---|
| Yoruba (`yor`) | 99.60% | 99.60% | 99.60% |
| Hausa (`hau`) | 99.60% | 99.20% | 99.40% |
| Igbo (`ibo`) | 98.79% | 98.20% | 98.50% |
| Nigerian Pidgin (`pcm`) | 99.20% | 98.80% | 99.00% |
| English (`eng`) | 97.63% | 99.00% | 98.31% |
| **Overall (Macro)** | | | **98.96%** |

**Average latency**: ~13.30 ms/sentence (CPU or GPU)

### Installation

```bash
pip install olaverse[deeplearning]
# installs: torch, transformers
```

### Usage via Olaverse

```python
from olaverse import LIDNeural5

detector = LIDNeural5()
detector.load()  # downloads olaverse/lid-neural-5 from Hugging Face

# Predict dominant language
lang = detector.predict("Kedu ka ị mere today?")
print(lang)  # → 'ibo'

# Get probability distribution over all 5 classes
probs = detector.predict_proba("How far, wetin dey happen?")
print(probs)
# → {'eng': 0.002, 'hau': 0.001, 'ibo': 0.003, 'pcm': 0.991, 'yor': 0.003}
```

### Usage via Hugging Face (direct)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("olaverse/lid-neural-5")
model = AutoModelForSequenceClassification.from_pretrained("olaverse/lid-neural-5")

lid = pipeline("text-classification", model=model, tokenizer=tokenizer)
result = lid("Bawo ni, se daadaa ni?")
print(result)  # → [{'label': 'yor', 'score': 0.9987}]
```

::: olaverse.llm.LIDNeural5
