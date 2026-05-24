# Olaverse
> Advanced ML infrastructure and production-ready interface to load and run all Olaverse models.

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-CPU%20&%20GPU-orange)](https://github.com/)

Standard NLP tools and model interfaces don't capture local nuances, custom tokenizers, or specific fine-tuned downstream configurations (like contract analysis & legal reasoning). 

**Olaverse** is a unified Python package and developer interface designed to run all Olaverse model families—ranging from lightweight CPU-only local NLP modules to large enterprise-grade legal and reasoning models.

---

## Key Features

- **🏛️ Enterprise Legal AI**: Built-in support and inference pipeline for `olaverse/legal-peace-v1.0` (fine-tuned Mistral-7B for contract analysis and legal reasoning).
- **🚀 GPU & CPU Optimized**: Lazy-loading and custom CUDA/Unsloth optimization layers for large LLMs alongside hyper-efficient, 100% offline local NLP tools.
- **📦 Optimized Tokenizers (`OTK-BPE`)**: Custom Byte-Level BPE tokenizers (e.g. `olaverse/otk-bpe-50k`) trained on dedicated Nigerian and African language corpora (up to **63% fewer tokens** compared to GPT-4).
- **🗣️ Advanced Diacritic Restoration**:
  - Yoruba Diacritizer (dot-below only): **97.5% character accuracy**.
  - Yoruba Diacritizer (full tonal): **90.0% word accuracy** via Viterbi decoding.
  - Igbo Diacritizer: **95.2% character accuracy**.
- **🎭 Context-Aware Sentiment Analysis**: Captures sentiment nuances in Pidgin English and regional languages (**72% accuracy**).
- **🔒 Nigerian-specific PII Masking**: Automatically masks emails, local +234/080 phone formats, BVN, and NIN.
- **🇳🇬 Robust Language Detection**: Accurately classifies text across 5 languages: Yoruba (`yor`), Hausa (`hau`), Igbo (`ibo`), Pidgin (`pcm`), and English (`eng`) with **98.12% accuracy** (LIDLite5) and **98.96% accuracy** (LIDNeural5).

---

## Installation

Install core library (lightweight, CPU offline-first tokenizers & text helpers):

```bash
pip install olaverse
```

Install with transformer model dependencies (torch + transformers, for `LIDNeural5`):

```bash
pip install olaverse[deeplearning]
```

Install with legal model dependencies (unsloth, torch, etc. for GPU inference):

```bash
pip install olaverse[legal]
```

For development mode (with Jupyter notebooks and training dependencies):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Legal & Contract Analysis (`LegalPeace`)
Direct interface for loading and running the `olaverse/legal-peace-v1.0` model. Uses Unsloth under the hood for fast, memory-efficient inference.
```python
from olaverse.llm import LegalPeace

# Instantiates the wrapper with default configurations (or enter custom model name/parameters)
model = LegalPeace(model_name="olaverse/legal-peace-v1.0")

# Loads model & tokenizer lazily (requires GPU and unsloth installed)
model.load()


prompt = "Analyze this contract clause: 'The parties agree that all disputes shall be resolved through binding arbitration in Delaware.' What are the key legal implications?"
response = model.generate(prompt, max_new_tokens=300, temperature=0.7)
print(response)
```

### 2. Custom Tokenizers (`Tokenizer`)
Use optimized Byte-Level BPE tokenizers without needing local `.json` file paths. Loads from Hugging Face Hub automatically if not cached. For details on performance and training, see the [otk-bpe repository](https://github.com/Olaverse-Labs/otk-bpe).
```python
from olaverse.nlp import Tokenizer

# Load the 50k unified model
tok = Tokenizer("naija") 
# Or specific languages (e.g., "yo", "ig", "ha", "pcm", or full model name "otk-bpe-50k-yo")
# tok = Tokenizer("yo")

tokens = tok.encode("Ẹ kú àbọ̀")
print(tokens) # → [124, 381]

decoded = tok.decode(tokens)
print(decoded) # → "Ẹ kú àbọ̀"
```

### 3. Language Detection (`LIDLite5` & `LIDNeural5`)
Identify whether text is Yoruba, Hausa, Igbo, Nigerian Pidgin, or English.

#### Option A: Lightweight (Zero-Dependency CPU)
```python
from olaverse import LIDLite5

detector = LIDLite5()
print(detector.predict("How far, wetin dey happen?")) # → 'pcm'

# Get confidence scores across all 5 classes
probs = detector.predict_proba("How far, wetin dey happen?")
print(probs) # → {'eng': 0.0006, 'hau': 0.0014, ...}
```

#### Option B: Neural (Transformer GPU/CPU)
```python
from olaverse import LIDNeural5

detector = LIDNeural5()
detector.load()  # Downloads and caches model from HF (olaverse/lid-neural-5)
                 # Requires: pip install olaverse[deeplearning]

print(detector.predict("How far, wetin dey happen?")) # → 'pcm'
```

### 4. Yoruba & Igbo Diacritizer
Restore missing diacritics in Yoruba or Igbo text.
```python
from olaverse.nlp import diacritize_yoruba, diacritize_yoruba_dot_below, diacritize_igbo

# Dot-below only (no tones)
diacritize_yoruba_dot_below("Ojo lo si oja")
# → 'Ọjọ lo si ọja'

# Full tonal diacritics
diacritize_yoruba("Ojo lo si oja lana")
# → 'Ọjọ́ ló sí ọjà lànà'

# Igbo diacritics
diacritize_igbo("Kedu ka i mere")
# → 'Kedụ ka ị mere'
```

### 5. Sentiment Analysis
Analyze sentiment across English and Nigerian languages.
```python
from olaverse.nlp import analyze_sentiment

analyze_sentiment("This film too sweet!")
# → {'label': 'positive', 'confidence': 0.74}

analyze_sentiment("I no like am at all")
# → {'label': 'negative', 'confidence': 0.68}
```

### 6. Text Preprocessing & PII Masking
Strip or mask sensitive data (NIN, BVN, phone numbers) while preserving Pidgin particles like "sha", "sef", "abeg".
```python
from olaverse.nlp import mask_pii, is_pidgin_particle

mask_pii("Call me on 08012345678 or my BVN is 22233344455")
# → 'Call me on [PHONE] or my BVN is [BVN]'

is_pidgin_particle("sha")   # → True
is_pidgin_particle("sef")   # → True
```

### 7. Constants & Local Helpers
```python
from olaverse.utils.constants import STATES, BANKS, format_naira, get_telco

STATES["Lagos"]              # → 'Ikeja'
BANKS["Guaranty Trust Bank"]  # → '058'
format_naira(1500000)        # → '₦1,500,000.00'
get_telco("08031234567")     # → 'MTN'
```

---

## Design Philosophy

1. **Unified Interface**: A single, clean library to access all Olaverse NLP, Tokenizer, and Large Language Models.
2. **Resource-Adaptive**: Keeps local CPU helpers lightweight and fully offline-capable, while offering scalable legal and reasoning GPU wrappers.
3. **Honest Metrics & Benchmarks**: We believe in sharing open, verifiable performance indicators of our tokenizers and classifiers against global baselines.

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
