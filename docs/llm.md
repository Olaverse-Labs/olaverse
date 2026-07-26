# Language Models

The `olaverse.llm` module provides clean interfaces for running transformer-based language models — with correct generation defaults, stop tokens, and endpoint flexibility built in, so you don't have to figure them out yourself.

---

## MIST — General-Purpose Model Family

The MIST family is olaverse's flagship LLM series, built by blending the best Llama 3.1 models via **DARE+TIES** and **Frankenmerge** techniques.

**Model Cards**: [MIST-Mini-8B](https://huggingface.co/olaverse/MIST-Mini-8B) · [MIST-1-70B](https://huggingface.co/olaverse/MIST-1-70B) · [MIST-1-140B](https://huggingface.co/olaverse/MIST-1-140B) · [MIST-1-140B-4bit](https://huggingface.co/olaverse/MIST-1-140B-4bit) · [MIST-Mini-8B-Thinking](https://huggingface.co/olaverse/MIST-Mini-8B-Thinking)

### Model Variants

| `size=` | Model | Params | Speed | Best for |
|---|---|---|---|---|
| `"8b"` / `"mini"` | MIST-Mini-8B | 8B | ~63 tok/s | Fast everyday use |
| `"70b"` | MIST-1-70B | 70B | ~23 tok/s | Structured, detailed output |
| `"140b"` | MIST-1-140B | 140B | ~8 tok/s | Deepest reasoning |
| `"140b-4bit"` | MIST-1-140B-4bit | 140B (4-bit) | ~8 tok/s | Single H100/H200 (70GB VRAM) |
| `"thinking"` | MIST-Mini-8B-Thinking | 8B | ~55 tok/s | Step-by-step reasoning with `<think>` |

### Why use the wrapper?

A bare `from_pretrained` call on MIST will produce rambling or cut-off output because:

- **Stop tokens differ per variant.** MIST-8B/Thinking inherited ChatML `<|im_end|>` (token `128040`) from its DARE+TIES parents alongside Llama 3.1's native tokens. Omitting it causes the model to not stop cleanly. MIST-70B/140B use a different set — no ChatML.
- **`repetition_penalty` and `min_p` are required.** Without them, the model repeats and doesn't terminate. These values are verified; the defaults vary per variant.
- **The endpoint switch.** Same `.generate()` / `.chat()` API whether you're running locally or via Featherless, Modal, or your own vLLM server.

### Installation

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

### Usage — Local

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
print(model.generate("Write a Python retry decorator with exponential backoff."))
```

### Usage — Hosted (Featherless)

No GPU required. Create a free API key at [featherless.ai](https://featherless.ai).

```python
import os
from olaverse import MIST

model = MIST(
    size="70b",
    endpoint="featherless",
    api_key=os.environ["FEATHERLESS_API_KEY"],
)
print(model.generate("Summarise the key differences between 70B and 140B MIST models."))
```

### Usage — Hosted (Modal / custom vLLM)

```python
from olaverse import MIST

model = MIST(
    size="140b",
    endpoint="https://your-modal-endpoint.modal.run",
)
print(model.generate("Solve step by step: If 3x + 7 = 22, find x."))
```

### Multi-turn Chat

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

### Reasoning Variant

`MIST-Mini-8B-Thinking` was trained with 4 phases of GRPO reinforcement learning to show its reasoning before answering. The system prompt is set automatically.

```python
model = MIST(size="thinking")
model.load()

# Default system prompt already instructs the model to use <think> tags
response = model.generate("If a train travels 120 miles in 2 hours, what is its speed?")
# Response shows <think>...</think> then the final answer
```

### Hardware Requirements

| Variant | Precision | VRAM |
|---|---|---|
| 8B / Thinking | bfloat16 | 16 GB (RTX 3090/4090) |
| 8B / Thinking | 4-bit NF4 | 6 GB (RTX 3060+) |
| 70B | bfloat16 | 140 GB (1× H200 or 2× H100) |
| 70B | 4-bit NF4 | 40 GB (1× A100/H100) |
| 140B | bfloat16 | 280 GB (2× H200) |
| 140B | 4-bit NF4 | 70 GB (1× H200) |

::: olaverse.llm.MIST

---

## LegalPeace — Legal Contract Reasoning

!!! warning "Beta Model"
    LegalPeace is a **research/beta model**. Always verify outputs with a qualified legal professional. Trained primarily on U.S. legal data.

`LegalPeace` is a fine-tuned **Mistral-7B-v0.3** for contract analysis and legal reasoning, loaded via `unsloth` for fast 4-bit quantized inference.

**Model Card**: [olaverse/legal-peace-v1.0](https://huggingface.co/olaverse/legal-peace-v1.0)

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
pip install olaverse[legal]
# or: pip install unsloth
```

### Usage

```python
from olaverse import LegalPeace

model = LegalPeace()
model.load()  # requires GPU + unsloth

clause = """
Analyze this clause: 'All disputes shall be resolved through binding
arbitration in Delaware.' What are the key implications?
"""
print(model.generate(clause, max_new_tokens=300))
```

### Supported Use Cases

- Contract clause analysis and risk flagging
- Legal research assistance
- Evidence evaluation
- Case outcome prediction
- Legal Q&A

::: olaverse.llm.LegalPeace

---

## LIDNeural5 — Neural Language Identification

!!! tip "Better imported from `olaverse.nlp`"
    `LIDNeural5` is a sequence classifier, not an LLM — its natural home is `olaverse.nlp`.
    Both `from olaverse.nlp import LIDNeural5` and `from olaverse.llm import LIDNeural5` work.

See the **[NLP & Tokenization](nlp.md#lidneural5)** page for full documentation and examples.

::: olaverse.llm.LIDNeural5

---

## MISTTitleGenerator — Chat Titles

**Model Card**: [olaverse/mist-tg-0.3b](https://huggingface.co/olaverse/mist-tg-0.3b)

Generates a short title from a user's first message — the "name this conversation"
step in a chat UI. Byte-level seq2seq (~300M), no prompt template needed.

```bash
pip install olaverse[deeplearning]
```

```python
from olaverse import MISTTitleGenerator

titler = MISTTitleGenerator()
titler.generate("My laptop keeps freezing every time I open more than five browser tabs, any idea why?")
# → 'Laptop Freezing Impact'

# One forward pass for many messages
titler.generate_batch(["How do I center a div in CSS?", "What makes Yoruba tonal?"])
# → ['CSS Centering Div In CSS', 'Yoruba Tonal Language']
```

Input is truncated to 256 bytes — the model's trained input length. Pass the
user's first message raw; no language tag or instruction prefix.

!!! warning "English-first; Latin script only"
    Trained on English chat messages. Latin-script languages often work as a
    byte-level transfer side effect, but are not guaranteed. Non-Latin scripts
    (CJK, Hangul, Devanagari, Ethiopic) are **not** supported — the model emits
    unrelated English text rather than failing loudly.

::: olaverse.llm.MISTTitleGenerator

---

## MISTQuestionGenerator — Question Generation

**Model Card**: [olaverse/mist-qg-1.5b](https://huggingface.co/olaverse/mist-qg-1.5b)

Given a passage, generates natural search-style questions that the passage
directly answers — as a question-generation endpoint, or as a data factory for
minting `(query, positive)` pairs to train retrievers and rerankers (pairs well
with [`Reranker` and `Embedder`](nlp.md#retrieval-new-in-v015)).

```bash
pip install olaverse[deeplearning]
```

```python
from olaverse import MISTQuestionGenerator

qg = MISTQuestionGenerator()

passage = ("Tides are caused by the gravitational pull of the moon and, "
           "to a lesser extent, the sun, acting on Earth's oceans.")

qg.generate(passage)
# → ['What causes tides according to the passage?',
#    'Is the gravitational pull from the sun as significant as it is from the moon...?',
#    'How do tides differ between the moon and the sun?']

# A French passage → French questions
qg.generate(french_passage, n=3, language="fra")   # or "fr", or "French"
# → ["Qu'est-ce qui cause les marées?", ...]
```

!!! info "`language=` names the passage's language"
    This is **same-language** question generation, not translation. `language=`
    tells the model what language the passage is written in, and questions come
    back in that language. Pointing it at an English passage and asking for
    Yoruba does not produce usable Yoruba — pass a Yoruba passage instead.

`language=` accepts an ISO 639-3 code (`"fra"`), an ISO 639-1 code (`"fr"`), or
the English name (`"French"`) — all three resolve identically. The model returns
strict JSON internally; the wrapper parses it and hands back a plain `list[str]`,
falling back to string recovery if a long generation gets truncated mid-JSON.
Raise `max_new_tokens` (default 250) for large `n`.

**Supported languages (25)**: `eng, fra, deu, spa, por, ita, nld, rus, pol, tur, vie, ind, hin, jpn, kor, yor, ibo, hau, swh, amh, zul, xho, sna, som, afr` — also available programmatically as `olaverse.QG_LANGUAGES`.

The wrapper uses the teacher prompt this model was distilled with, and sizes the
JSON skeleton to `n` so the model returns the number of questions you asked for.

Three guards fire automatically:

- an empty passage raises `ValueError`
- a passage under 20 characters warns — the model needs paragraph-length input
- `amh`, `som`, and `sna` warn as the model card's lower-confidence languages
  (exposed as `olaverse.llm.mist_tasks.QG_WEAK_LANGUAGES`)

::: olaverse.llm.MISTQuestionGenerator
