# Large Language Models

The `olaverse.llm` module provides interfaces for running computationally heavy transformer-based models, such as LLMs and deep neural detectors.

## Legal AI

The `LegalPeace` interface makes it trivial to load the `olaverse/legal-peace-v1.0` model. It utilizes `unsloth` for high-performance memory-efficient loading on GPUs.

::: olaverse.llm.LegalPeace

## Neural Language Detection

For maximum accuracy (98.96%), the `LIDNeural5` uses a full Hugging Face transformer backend to classify Yoruba, Hausa, Igbo, Pidgin, and English.

::: olaverse.llm.LIDNeural5
