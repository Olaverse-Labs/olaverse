![Olaverse — small models, sharp focus: LID, DiacNet, MIST, Prism](docs/assets/banner.png)

# Olaverse Documentation

[![PyPI Version](https://img.shields.io/pypi/v/olaverse.svg)](https://pypi.org/project/olaverse/)
[![Downloads](https://img.shields.io/pypi/dm/olaverse.svg)](https://pypi.org/project/olaverse/)
[![Python Version](https://img.shields.io/pypi/pyversions/olaverse.svg)](https://pypi.org/project/olaverse/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-olaverse-yellow)](https://huggingface.co/olaverse)
[![Platform](https://img.shields.io/badge/platform-CPU%20&%20GPU-orange)](https://github.com/Olaverse-Labs/olaverse)
[![Docs](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://Olaverse-Labs.github.io/olaverse/)

Welcome to the official developer documentation for the **Olaverse SDK**.

**Olaverse** is an open-source multilingual AI infrastructure toolkit for building NLP, speech, retrieval, and language systems for underrepresented languages.

## 30-Second Quick Start

```bash
pip install olaverse
```

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="auto")       # detects the language, routes to the right model
d.restore("Ojo lo si oja lana")     # → 'Òjó lọ sí ọjà lana'

# 10 languages via the multilingual model (pip install olaverse[deeplearning]):
d = Diacritizer(model="diacnet-1.0", lang="yo")
d.restore("se eranko naa si gbo o?")   # → 'ṣé ẹranko náà sì gbọ́ ọ?'
```

New in v0.2.0 — title and question generation (`pip install olaverse[deeplearning]`):

```python
from olaverse import MISTTitleGenerator, MISTQuestionGenerator

MISTTitleGenerator().generate("My laptop keeps freezing when I open too many tabs, why?")
# → 'Laptop Freezing Impact'

MISTQuestionGenerator().generate(passage, n=3, language="eng")
# → ['What causes ocean tides?', 'Does the sun affect tides?', ...]
```

**📚 Full API Documentation: [https://Olaverse-Labs.github.io/olaverse/](https://Olaverse-Labs.github.io/olaverse/)**
**📦 PyPI: [https://pypi.org/project/olaverse/](https://pypi.org/project/olaverse/)**

---

## Key Capabilities

- **🗣️ Natural Language Processing**: Diacritization for 10+ languages (Yoruba, Igbo, Hausa, Vietnamese, Polish, Turkish, Portuguese, Spanish, French, Italian via `diacnet-1.0`), Language Detection from 5 to 25 languages (`LIDLite5`/`LIDNeural5`, `LIDLite25`/`LIDNeural25`, and the Nigerian-only `LIDNeural5_1`), Byte-Level BPE tokenization (Nigerian languages plus Swahili/Kinyarwanda/merged families), PII masking, and TTS text normalization.
- **⚡ MIST Model Family**: Unified interface for the MIST LLM family (8B, 70B, 140B, Thinking). Supports local inference via `transformers` and hosted inference via Featherless or any OpenAI-compatible endpoint. Correct stop tokens and generation defaults per variant are baked in. Plus two task-specific models: `MISTTitleGenerator` (short chat titles from a user's first message) and `MISTQuestionGenerator` (search-style question generation from a passage, across 25 languages).
- **🧠 Domain LLMs**: `LegalPeace` — memory-efficient 4-bit inference for legal contract reasoning (fine-tuned Mistral-7B-v0.3).
- **🔎 Retrieval**: `Reranker` (cross-encoder, RAG/search second stage) and `Embedder` (cross-lingual Hausa/Yoruba/Igbo sentence embeddings).
- **🖼️ Vision — Prism**: `PrismUpscaler` (2x/4x/arbitrary-resolution super-resolution), `PrismDenoiser` (noise/blur/compression removal), and `PrismSteganography` (hide/recover short messages in images).
- **📊 Datasets**: `load_dataset` / `list_datasets` — direct access to every public olaverse dataset on Hugging Face (reranker training pairs, multilingual QG passages, DiacBench, and more).
- **🎙️ Speech Architecture** *(Roadmap / Experimental)*: TTS pipeline architecture connecting normalization, diacritization, acoustic model, and vocoder. The NLP front-end is production-ready; acoustic synthesis is in development.
- **🌍 Global Utilities**: Currency formatters, generic constants, and `.wav` audio I/O tools.

---

## Quick Install

```bash
# Core (NLP, tokenizer, lightweight LID)
pip install olaverse

# Neural models (LIDNeural5/25/5_1, diacnet-1.0, MIST local inference)
pip install olaverse[deeplearning]

# Lightweight 25-language LID (fastText, CPU-only)
pip install olaverse[lid]

# Retrieval (Reranker, Embedder)
pip install olaverse[retrieval]

# Vision (PrismUpscaler, PrismDenoiser, PrismSteganography)
pip install olaverse[vision]

# Hosted inference (MIST via Featherless, Modal, etc.)
pip install olaverse[hosted]

# Legal reasoning (LegalPeace)
pip install olaverse[legal]

# Datasets (load_dataset — reranker pairs, QG passages, DiacBench, ...)
pip install olaverse[data]
```

---

## Navigation

- **[Models](https://Olaverse-Labs.github.io/olaverse/models/)**: Product pages for every model family — DiacNet, LID, OTK-BPE, Retrieval, MIST, LegalPeace, Prism — with comparison tables.
- **[Benchmarks](https://Olaverse-Labs.github.io/olaverse/benchmarks/)**: All published numbers in one place.
- **[Solutions](https://Olaverse-Labs.github.io/olaverse/solutions/)**: Worked pipelines — Speech AI, OCR, search, education, support, translation.
- **[NLP & Tokenization](https://Olaverse-Labs.github.io/olaverse/nlp/)**: `Tokenizer`, Language Detection, Diacritization, Retrieval (`Reranker`/`Embedder`), PII masking, TTS normalizer.
- **[Language Models](https://Olaverse-Labs.github.io/olaverse/llm/)**: `MIST` model family, `MISTTitleGenerator`, `MISTQuestionGenerator`, `LegalPeace`, `LIDNeural5`.
- **[Vision](https://Olaverse-Labs.github.io/olaverse/vision/)**: `PrismUpscaler`, `PrismDenoiser`, `PrismSteganography`.
- **[Datasets](https://Olaverse-Labs.github.io/olaverse/datasets/)**: `load_dataset`, `list_datasets`, `dataset_info` — all public olaverse datasets.
- **[Speech Architecture](https://Olaverse-Labs.github.io/olaverse/speech/)**: `TTSPipeline` and base classes (experimental — roadmap).
- **[Global Utilities](https://Olaverse-Labs.github.io/olaverse/utils/)**: Constants and audio utilities.
- **[Enterprise](https://Olaverse-Labs.github.io/olaverse/enterprise/)**: Commercial support — fine-tuning, custom datasets, deployment.
- **[Roadmap](https://Olaverse-Labs.github.io/olaverse/roadmap/)**: What's shipped and what's next.
