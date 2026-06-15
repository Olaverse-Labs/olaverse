# Olaverse Documentation

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.4-blue)](https://pypi.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-CPU%20&%20GPU-orange)](https://github.com/)
[![Docs](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://Olaverse-Labs.github.io/olaverse/)

Welcome to the official developer documentation for the **Olaverse SDK**.

**Olaverse** is a unified Python package and developer interface for African NLP, Large Language Models, and Text-to-Speech architecture.

**📚 Full API Documentation: [https://Olaverse-Labs.github.io/olaverse/](https://Olaverse-Labs.github.io/olaverse/)**

---

## Key Capabilities

- **🗣️ Natural Language Processing**: Diacritization (Yoruba, Igbo), Language Detection for 5 Nigerian languages (lightweight `LIDLite5` and neural `LIDNeural5`), Byte-Level BPE tokenization, PII masking, and TTS text normalization.
- **⚡ MIST Model Family**: Unified interface for the MIST LLM family (8B, 70B, 140B, Thinking). Supports local inference via `transformers` and hosted inference via Featherless or any OpenAI-compatible endpoint. Correct stop tokens and generation defaults per variant are baked in.
- **🧠 Domain LLMs**: `LegalPeace` — memory-efficient 4-bit inference for legal contract reasoning (fine-tuned Mistral-7B-v0.3).
- **🎙️ Speech Architecture** *(Roadmap / Experimental)*: TTS pipeline architecture connecting normalization, diacritization, acoustic model, and vocoder. The NLP front-end is production-ready; acoustic synthesis is in development.
- **🌍 Global Utilities**: Nigerian currency formatters, generic constants, and `.wav` audio I/O tools.

---

## Quick Install

```bash
# Core (NLP, tokenizer, lightweight LID)
pip install olaverse

# Neural models (LIDNeural5, MIST local inference)
pip install olaverse[deeplearning]

# Hosted inference (MIST via Featherless, Modal, etc.)
pip install olaverse[hosted]

# Legal reasoning (LegalPeace)
pip install olaverse[legal]
```

---

## Navigation

- **[NLP & Tokenization](https://Olaverse-Labs.github.io/olaverse/nlp/)**: `Tokenizer`, Language Detection, Diacritization, PII masking, TTS normalizer.
- **[Language Models](https://Olaverse-Labs.github.io/olaverse/llm/)**: `MIST` model family, `LegalPeace`, `LIDNeural5`.
- **[Speech Architecture](https://Olaverse-Labs.github.io/olaverse/speech/)**: `TTSPipeline` and base classes (experimental — roadmap).
- **[Global Utilities](https://Olaverse-Labs.github.io/olaverse/utils/)**: Constants and audio utilities.
