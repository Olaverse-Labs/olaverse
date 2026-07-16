![Olaverse — small models, sharp focus: LID, DiacNet, MIST, Prism](docs/assets/banner.png)

# Olaverse Documentation

[![PyPI Version](https://img.shields.io/pypi/v/olaverse.svg)](https://pypi.org/project/olaverse/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-CPU%20&%20GPU-orange)](https://github.com/Olaverse-Labs/olaverse)
[![Docs](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://Olaverse-Labs.github.io/olaverse/)

Welcome to the official developer documentation for the **Olaverse SDK**.

**Olaverse** is a unified Python package and developer interface for African NLP, Large Language Models, and Text-to-Speech architecture.

**📚 Full API Documentation: [https://Olaverse-Labs.github.io/olaverse/](https://Olaverse-Labs.github.io/olaverse/)**
**📦 PyPI: [https://pypi.org/project/olaverse/](https://pypi.org/project/olaverse/)**

---

## Key Capabilities

- **🗣️ Natural Language Processing**: Diacritization for 10+ languages (Yoruba, Igbo, Hausa, Vietnamese, Polish, Turkish, Portuguese, Spanish, French, Italian via `diacnet-1.0`), Language Detection from 5 to 25 languages (`LIDLite5`/`LIDNeural5`, `LIDLite25`/`LIDNeural25`, and the Nigerian-only `LIDNeural5_1`), Byte-Level BPE tokenization (Nigerian languages plus Swahili/Kinyarwanda/merged families), PII masking, and TTS text normalization.
- **⚡ MIST Model Family**: Unified interface for the MIST LLM family (8B, 70B, 140B, Thinking). Supports local inference via `transformers` and hosted inference via Featherless or any OpenAI-compatible endpoint. Correct stop tokens and generation defaults per variant are baked in.
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

- **[NLP & Tokenization](https://Olaverse-Labs.github.io/olaverse/nlp/)**: `Tokenizer`, Language Detection, Diacritization, Retrieval (`Reranker`/`Embedder`), PII masking, TTS normalizer.
- **[Language Models](https://Olaverse-Labs.github.io/olaverse/llm/)**: `MIST` model family, `LegalPeace`, `LIDNeural5`.
- **[Vision](https://Olaverse-Labs.github.io/olaverse/vision/)**: `PrismUpscaler`, `PrismDenoiser`, `PrismSteganography`.
- **[Datasets](https://Olaverse-Labs.github.io/olaverse/datasets/)**: `load_dataset`, `list_datasets`, `dataset_info` — all public olaverse datasets.
- **[Speech Architecture](https://Olaverse-Labs.github.io/olaverse/speech/)**: `TTSPipeline` and base classes (experimental — roadmap).
- **[Global Utilities](https://Olaverse-Labs.github.io/olaverse/utils/)**: Constants and audio utilities.
