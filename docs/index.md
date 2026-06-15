<div class="ov-hero">
  <div class="ov-hero-badge">v0.1.4 — Now with MIST</div>
  <h1 class="ov-hero-title">The Olaverse SDK</h1>
  <p class="ov-hero-sub">African NLP · MIST Language Models · Text-to-Speech Architecture</p>
  <div class="ov-hero-install">
    <span class="ov-hero-install-label">pip install olaverse</span>
  </div>
  <div class="ov-hero-links">
    <a href="nlp/" class="md-button md-button--primary">Get Started</a>
    <a href="https://github.com/Olaverse-Labs/olaverse" class="md-button" target="_blank">GitHub</a>
    <a href="https://huggingface.co/olaverse" class="md-button" target="_blank">Hugging Face</a>
  </div>
</div>

---

## What is Olaverse?

**Olaverse** is a unified Python SDK for African language AI. It gives you production-ready NLP tools, a clean interface to the MIST model family, domain-specific LLMs, and a TTS pipeline architecture — all in one package.

<div class="ov-grid">

<div class="ov-card">
  <div class="ov-card-icon">🗣️</div>
  <div class="ov-card-title">NLP & Tokenization</div>
  <div class="ov-card-body">Diacritization for Yoruba and Igbo, language detection for 5 Nigerian languages, Byte-Level BPE tokenizers with 0% OOV, PII masking, and TTS text normalization.</div>
  <a href="nlp/" class="ov-card-link">Explore NLP →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">⚡</div>
  <div class="ov-card-title">MIST Model Family</div>
  <div class="ov-card-body">8B, 70B, 140B, and Thinking variants. Correct stop tokens and sampling defaults per variant baked in. Local via <code>transformers</code> or hosted via Featherless/Modal.</div>
  <a href="llm/" class="ov-card-link">Explore MIST →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">🧠</div>
  <div class="ov-card-title">Domain LLMs</div>
  <div class="ov-card-body"><code>LegalPeace</code> — fine-tuned Mistral-7B for contract analysis and legal reasoning. Memory-efficient 4-bit inference via unsloth.</div>
  <a href="llm/#legalpace-legal-contract-reasoning-beta" class="ov-card-link">Explore LLMs →</a>
</div>

<div class="ov-card">
  <div class="ov-card-icon">🎙️</div>
  <div class="ov-card-title">Speech Architecture</div>
  <div class="ov-card-body">TTS pipeline connecting normalization, diacritization, acoustic model, and vocoder. The NLP front-end is production-ready. Acoustic synthesis is on the roadmap.</div>
  <a href="speech/" class="ov-card-link">Explore Speech →</a>
</div>

</div>

---

## Install

=== "Core"
    ```bash
    pip install olaverse
    ```
    Includes: NLP tools, diacritizers, tokenizers, lightweight LID, PII masking.

=== "Neural Models"
    ```bash
    pip install olaverse[deeplearning]
    ```
    Adds: `LIDNeural5`, MIST local inference (requires GPU).

=== "Hosted Inference"
    ```bash
    pip install olaverse[hosted]
    ```
    Adds: MIST via Featherless, Modal, or any OpenAI-compatible endpoint.

=== "Legal"
    ```bash
    pip install olaverse[legal]
    ```
    Adds: `LegalPeace` contract analysis model (requires GPU + unsloth).

=== "Everything"
    ```bash
    pip install olaverse[deeplearning,hosted,legal]
    ```

---

## Quick Start

### Language Detection

```python
from olaverse import LIDLite5, LIDNeural5

# Lightweight — zero GPU, instant (0.014 ms/sentence)
detector = LIDLite5()
detector.predict("Bawo ni, se daadaa ni?")  # → 'yor'

# Neural — 98.96% macro accuracy
neural = LIDNeural5()
neural.load()  # downloads olaverse/lid-neural-5 once
neural.predict("Kedu ka ị mere today?")  # → 'ibo'
neural.predict_proba("How far, wetin dey happen?")
# → {'pcm': 0.991, 'eng': 0.002, ...}
```

### Diacritization

```python
from olaverse import diacritize_yoruba, diacritize_igbo

diacritize_yoruba("Ojo lo si oja lana")
# → 'Ọjọ́ ló sí ọjà lànà'

diacritize_igbo("Kedu ka i mere")
# → 'Kedụ ka ị mere'
```

### MIST — Fast (8B, local)

```python
from olaverse import MIST

model = MIST(size="8b")
model.load()
print(model.generate("Explain what makes Yoruba a tonal language."))
```

### MIST — Hosted (70B via Featherless)

```python
import os
from olaverse import MIST

model = MIST(size="70b", endpoint="featherless", api_key=os.environ["FEATHERLESS_API_KEY"])
print(model.generate("Write a Python retry decorator with exponential backoff."))
```

### Tokenization

```python
from olaverse import Tokenizer

tok = Tokenizer("yo")  # Yoruba — 63% fewer tokens than GPT-4
ids = tok.encode("Ẹ kú àbọ̀")
tok.decode(ids)  # → 'Ẹ kú àbọ̀'
```

---

## Supported Languages

<div class="ov-lang-row">
  <span class="ov-lang-badge ov-lang-yor">Yoruba <code>yor</code></span>
  <span class="ov-lang-badge ov-lang-ibo">Igbo <code>ibo</code></span>
  <span class="ov-lang-badge ov-lang-hau">Hausa <code>hau</code></span>
  <span class="ov-lang-badge ov-lang-pcm">Nigerian Pidgin <code>pcm</code></span>
  <span class="ov-lang-badge ov-lang-eng">English <code>eng</code></span>
</div>

| Feature | yor | ibo | hau | pcm | eng |
|---|:---:|:---:|:---:|:---:|:---:|
| Language Detection (LIDLite5 / LIDNeural5) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Diacritization | ✅ | ✅ | — | — | — |
| BPE Tokenizer | ✅ | ✅ | ✅ | ✅ | via naija |
| TTS Normalization | ✅ | ✅ | — | — | — |

---

## Model Index

<div class="ov-model-table">

| Model | Task | Size | Speed | Install |
|---|---|---|---|---|
| `LIDLite5` | Language ID | 1.1 MB JSON | 0.014 ms | `olaverse` |
| `LIDNeural5` | Language ID | 484 MB | 13 ms | `olaverse[deeplearning]` |
| `MIST-Mini-8B` | General LLM | 15 GB | ~63 tok/s | `olaverse[deeplearning]` |
| `MIST-1-70B` | General LLM | 132 GB | ~23 tok/s | hosted or multi-GPU |
| `MIST-1-140B` | General LLM | 256 GB | ~8 tok/s | hosted or 2× H200 |
| `MIST-Mini-8B-Thinking` | Reasoning LLM | 15 GB | ~55 tok/s | `olaverse[deeplearning]` |
| `LegalPeace` | Legal reasoning | 7B (4-bit) | — | `olaverse[legal]` |
| `DiacNet` (5 variants) | Diacritization | 1 MB – 503 MB | — | `olaverse` / `[deeplearning]` |
| `OTK-BPE-50k` (5 variants) | Tokenization | ~3 MB each | — | `olaverse` |

</div>

---

## What's New in v0.1.4

- **`MIST` wrapper** — unified interface for all MIST variants with correct stop tokens, sampling defaults, and local/hosted endpoint switching
- **`LIDNeural5` moved to `olaverse.nlp`** — its correct home alongside `LIDLite5` (backward-compat import from `olaverse.llm` preserved)
- **`ExperimentalWarning`** on speech classes — honest signalling that acoustic synthesis is not yet available
- **`olaverse[hosted]`** extra — `pip install olaverse[hosted]` for Featherless/Modal inference
