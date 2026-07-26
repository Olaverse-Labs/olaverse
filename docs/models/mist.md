# MIST — LLM Family

**General-purpose language models from 8B to 140B parameters, built by blending the best Llama 3.1 models via DARE+TIES and Frankenmerge — with correct generation defaults baked into the SDK.**

```python
from olaverse import MIST

model = MIST(size="8b")
model.load()
print(model.generate("Explain what makes Yoruba a tonal language."))
```

**Model Cards**: [MIST-Mini-8B](https://huggingface.co/olaverse/MIST-Mini-8B) · [MIST-1-70B](https://huggingface.co/olaverse/MIST-1-70B) · [MIST-1-140B](https://huggingface.co/olaverse/MIST-1-140B) · [MIST-1-140B-4bit](https://huggingface.co/olaverse/MIST-1-140B-4bit) · [MIST-Mini-8B-Thinking](https://huggingface.co/olaverse/MIST-Mini-8B-Thinking)

---

## Which variant?

| `size=` | Model | Params | Speed | Best for |
|---|---|---|---|---|
| `"8b"` / `"mini"` | MIST-Mini-8B | 8B | ~63 tok/s | Fast everyday use |
| `"70b"` | MIST-1-70B | 70B | ~23 tok/s | Structured, detailed output |
| `"140b"` | MIST-1-140B | 140B | ~8 tok/s | Deepest reasoning |
| `"140b-4bit"` | MIST-1-140B-4bit | 140B (4-bit) | ~8 tok/s | Single H100/H200 (70GB VRAM) |
| `"thinking"` | MIST-Mini-8B-Thinking | 8B | ~55 tok/s | Step-by-step reasoning with `<think>` |

---

## Why use the SDK wrapper?

A bare `from_pretrained` call on MIST produces rambling or cut-off output because:

- **Stop tokens differ per variant.** MIST-8B/Thinking inherited ChatML `<|im_end|>` (token `128040`) from its DARE+TIES parents alongside Llama 3.1's native tokens. MIST-70B/140B use a different set — no ChatML.
- **`repetition_penalty` and `min_p` are required.** Without them, the model repeats and doesn't terminate. The verified defaults vary per variant.
- **One API, any backend.** Same `.generate()` / `.chat()` whether you run locally or via Featherless, Modal, or your own vLLM server.

---

## Installation

=== "Local inference"
    ```bash
    pip install olaverse[deeplearning]
    # requires GPU (CUDA or MPS)
    ```

=== "Hosted inference"
    ```bash
    pip install olaverse[hosted]
    # works on any machine — no GPU needed
    ```

---

## Usage

### Local

```python
from olaverse import MIST

model = MIST(size="8b")
model.load()  # downloads from Hugging Face, cached after first run
print(model.generate("Explain what makes Yoruba a tonal language."))
```

**4-bit quantization** — runs MIST-8B on a 6 GB GPU:

```python
model = MIST(size="8b", quantize=True)
model.load()
```

### Hosted (Featherless)

No GPU required. Create a free API key at [featherless.ai](https://featherless.ai).

```python
import os
from olaverse import MIST

model = MIST(size="70b", endpoint="featherless", api_key=os.environ["FEATHERLESS_API_KEY"])
print(model.generate("Summarise the key differences between 70B and 140B MIST models."))
```

### Hosted (Modal / custom vLLM)

```python
model = MIST(size="140b", endpoint="https://your-modal-endpoint.modal.run")
```

### Multi-turn chat

```python
messages = [
    {"role": "user",      "content": "What is the capital of Nigeria?"},
    {"role": "assistant", "content": "The capital of Nigeria is Abuja."},
    {"role": "user",      "content": "What languages are spoken there?"},
]
print(model.chat(messages))
```

### Streaming (hosted only)

```python
model = MIST(size="8b", endpoint="featherless", api_key="...")
for chunk in model.generate("Tell me about Lagos.", stream=True):
    print(chunk, end="", flush=True)
```

### Reasoning variant

`MIST-Mini-8B-Thinking` was trained with 4 phases of GRPO reinforcement learning to show its reasoning before answering. The system prompt is set automatically.

```python
model = MIST(size="thinking")
model.load()
response = model.generate("If a train travels 120 miles in 2 hours, what is its speed?")
# Response shows <think>...</think> then the final answer
```

---

## Hardware Requirements

| Variant | Precision | VRAM |
|---|---|---|
| 8B / Thinking | bfloat16 | 16 GB (RTX 3090/4090) |
| 8B / Thinking | 4-bit NF4 | 6 GB (RTX 3060+) |
| 70B | bfloat16 | 140 GB (1× H200 or 2× H100) |
| 70B | 4-bit NF4 | 40 GB (1× A100/H100) |
| 140B | bfloat16 | 280 GB (2× H200) |
| 140B | 4-bit NF4 | 70 GB (1× H200) |

---

## Task-specific MIST models

Two small single-purpose models, each with its own SDK class:

### `MISTTitleGenerator` — chat titles

**Model Card**: [mist-tg-0.3b](https://huggingface.co/olaverse/mist-tg-0.3b) · ~300M, byte-level seq2seq

```python
from olaverse import MISTTitleGenerator

titler = MISTTitleGenerator()
titler.generate("My laptop keeps freezing when I open too many tabs, why?")
# → 'Laptop Freezing Impact'
```

English-trained; Latin-script languages often work, non-Latin scripts do not.

### `MISTQuestionGenerator` — question generation

**Model Card**: [mist-qg-1.5b](https://huggingface.co/olaverse/mist-qg-1.5b) · ~1.5B, 25 languages

```python
from olaverse import MISTQuestionGenerator

qg = MISTQuestionGenerator()
qg.generate("Tides are caused by the gravitational pull of the moon...", n=3, language="eng")
```

!!! info "`language=` names the passage's language"
    Same-language question generation, not translation — pass a French passage
    with `language="fra"`, a Yoruba passage with `language="yor"`. ISO 639-3,
    ISO 639-1, and English names all work.

Full reference for both: **[Language Models →](../llm.md#misttitlegenerator-chat-titles)**

---

## API Reference

Full class reference: [Language Models → MIST](../llm.md#mist-general-purpose-model-family)
