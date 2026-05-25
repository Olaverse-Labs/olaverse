# Olaverse Documentation

Welcome to the official developer documentation for the **Olaverse SDK**.

**Olaverse** is a unified Python package and developer interface designed to seamlessly integrate state-of-the-art African NLP features, Text-to-Speech (TTS) architecture, and Large Language Models (LLMs) into your applications.

## Key Capabilities

- **🗣️ Natural Language Processing**: Accurate text diacritization (Yoruba, Igbo), Language Detection (covering 5 local languages), custom Byte-Level BPE tokenization, and robust PII masking.
- **🎙️ Speech Synthesis**: End-to-End Text-to-Speech (TTS) pipelines to translate normalized text and restored tones into high-fidelity audio waveforms.
- **🧠 Large Language Models**: Easy, memory-efficient inference interfaces for loading advanced domain-specific LLMs (like `LegalPeace` for legal contract reasoning).
- **🌍 Global Utilities**: Built-in constants, universal currency formatters, and generic `.wav` audio I/O tools.

## Quick Install

```bash
pip install olaverse
```

## Navigation

- **[NLP & Tokenization](nlp.md)**: Explore the `Tokenizer`, Language Detection, Diacritization, and PII masking tools.
- **[Speech Synthesis](speech.md)**: Learn how to use the End-to-End `TTSPipeline` and extend it with custom Acoustic models.
- **[Language Models](llm.md)**: Explore how to run large models like `LegalPeace` and the Neural language detector.
- **[Global Utilities](utils.md)**: Check out the built-in generic constants and audio utilities.
