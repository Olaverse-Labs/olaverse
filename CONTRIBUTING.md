# Contributing to Olaverse

Thank you for your interest in contributing to **Olaverse**! We're building the foundational AI infrastructure for African languages, covering everything from tokenization and language detection to text diacritization and Text-to-Speech (TTS).

Whether you're fixing a bug, adding a new dataset, or introducing state-of-the-art models for a new African language, we welcome your contributions!

## 🚀 Getting Started

### 1. Set Up Your Environment

We recommend using a virtual environment. The project is managed using standard Python packaging.

```bash
# Clone the repository
git clone https://github.com/Olaverse-Labs/olaverse.git
cd olaverse

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the package in editable mode with development dependencies
pip install -e .
pip install pytest
```

### 2. Running Tests

Before submitting any code, ensure all unit tests pass. We use `pytest` for testing.

```bash
python -m pytest tests/ -v
```

If you add a new feature or model, **you must add corresponding tests** in the `tests/` directory.

## 🏗 Architecture & Code Organization

The SDK is organized into submodules:

* `olaverse/nlp/`: Contains the `Diacritizer`, `Tokenizer`, `LIDLite5`, and Text Normalization (`TTSNormalizer`).
* `olaverse/speech/`: Contains the foundational base classes and pipelines for Acoustic Models and Vocoders (`TTSPipeline`).
* `olaverse/llm/`: Contains Large Language Model logic and high-level detector models.

## 🧩 Adding a New Model

### NLP Models (e.g., Diacritizer)
If you have trained a new Diacritizer for an African language and uploaded it to Hugging Face:
1. Do **not** commit the raw `.pt` weights to the repository.
2. Update the `MODEL_REGISTRY` located inside `olaverse/nlp/diacritizer.py` to map a new Model ID to your architecture.
3. Add a unit test verifying its basic functionality in `tests/`.

### Speech Models (TTS/Acoustic/Vocoder)
If you are adding a new TTS model:
1. Create your model class inside `olaverse/speech/` and ensure it inherits from `BaseAcousticModel` or `BaseVocoder` (found in `olaverse/speech/base.py`).
2. Ensure the class implements the `load_weights()` and `forward()` / `generate()` abstract methods.
3. Update `olaverse/speech/__init__.py` to export your new model.

## 📝 Pull Request Process

1. **Fork the repo** and create your branch from `main`.
2. **Commit your changes**: Write clear, descriptive commit messages.
3. **Format your code**: Ensure it adheres to standard PEP8 formatting.
4. **Push to your fork** and submit a Pull Request.
5. Provide a clear description in your PR of what was changed and any results from training metrics if you are introducing a new model.

## 🤝 Code of Conduct
Please be respectful and collaborative. We are building this for the entire continent, and we welcome developers from all backgrounds!
