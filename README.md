# Olaverse 🇳🇬
> Sovereign ML Infrastructure and production-ready NLP tools for Nigeria.

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-CPU%20offline-orange)](https://github.com/)

Standard NLP tools don't work well for Nigerian languages. Global tokenizers strip Yoruba/Igbo diacritics, sentiment analyzers fail on Nigerian Pidgin, and PII masking libraries are completely unaware of local formats like BVN, NIN, and +234 phone formats.

**Olaverse** is an open-source Python library designed to solve this. Built for the real-world constraints of developing and deploying ML models in Nigeria: **works fully offline**, **runs on CPU**, requires **no GPU**, and is optimized for low-compute environments (runs smoothly on 4GB RAM).

---

## Key Features

- **🚀 CPU & Offline First**: Small models are bundled directly with the package; larger ones download once and cache locally.
- **🇳🇬 Robust Language Detection**: Accurately classifies text across 5 languages: Yoruba (`yor`), Hausa (`hau`), Igbo (`ibo`), Pidgin (`pcm`), and English (`eng`) with **96.6% accuracy**.
- **🗣️ Advanced Diacritic Restoration**:
  - Yoruba Diacritizer (dot-below only): **97.5% character accuracy**.
  - Yoruba Diacritizer (full tonal): **90.0% word accuracy** via Viterbi decoding.
  - Igbo Diacritizer: **95.2% character accuracy**.
- **🎭 Context-Aware Sentiment Analysis**: Captures sentiment nuances in Pidgin English and regional languages (**72% accuracy**).
- **🔒 Nigerian-specific PII Masking**: Automatically masks emails, local +234/080 phone formats, BVN, and NIN.
- **📦 Custom Tokenizers**: Custom BPE tokenizers trained on dedicated Nigerian language corpora (up to **63% fewer tokens** compared to GPT-4).
- **🗂️ Dataset Loaders**: Download and offline-cache popular Nigerian datasets (NaijaSenti, MasakhaNER, MasakhaNEWS, etc.) in a single line of code.

---

## Installation

Install via pip:

```bash
pip install olaverse
```

For development mode (with Jupyter notebooks and training dependencies):

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### 1. Language Detection
Detect if a sentence is Yoruba, Hausa, Igbo, Nigerian Pidgin, or English.
```python
from olaverse.nlp import detect_language

detect_language("Bawo ni, se daadaa ni?")   # → 'yor'
detect_language("Ina kwana?")                # → 'hau'
detect_language("Kedu ka ị mere?")           # → 'ibo'
detect_language("How far, wetin dey happen?") # → 'pcm'
```

### 2. Yoruba & Igbo Diacritizer
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

### 3. Sentiment Analysis
Analyze sentiment across English and Nigerian languages.
```python
from olaverse.nlp import analyze_sentiment

analyze_sentiment("This film too sweet!")
# → {'label': 'positive', 'confidence': 0.74}

analyze_sentiment("I no like am at all")
# → {'label': 'negative', 'confidence': 0.68}
```

### 4. Text Preprocessing & PII Masking
Strip or mask sensitive data (NIN, BVN, phone numbers) while preserving Pidgin particles like "sha", "sef", "abeg".
```python
from olaverse.nlp import mask_pii, is_pidgin_particle

mask_pii("Call me on 08012345678 or my BVN is 22233344455")
# → 'Call me on [PHONE] or my BVN is [BVN]'

is_pidgin_particle("sha")   # → True
is_pidgin_particle("sef")   # → True
```

### 5. Custom Tokenizers
Avoid splitting compounds or diacritics (like `ọ́` or `ẹ̀`).
```python
from olaverse.nlp import Tokenizer

tok = Tokenizer("naija")  # Unified model
tokens = tok.encode("Ẹ kú àbọ̀")
print(tokens)
# → [124, 381]
```

### 6. Nigerian Constants & Helpers
```python
from olaverse.utils.constants import STATES, BANKS, format_naira, get_telco

STATES["Lagos"]              # → 'Ikeja'
BANKS["Guaranty Trust Bank"]  # → '058'
format_naira(1500000)        # → '₦1,500,000.00'
get_telco("08031234567")     # → 'MTN'
```

---

## Model Metrics & Performance

| Model / Feature | Status | Accuracy / Metric | Size | Approach |
| :--- | :--- | :--- | :--- | :--- |
| **Language Detection** | ✅ | 96.6% accuracy | 2.5MB | Naive Bayes + Char N-grams (1-4) |
| **Yoruba Diacritizer (dot-below)** | ✅ | 97.5% char accuracy | 300KB | Syllabus dictionary lookup |
| **Yoruba Diacritizer (full)** | ✅ | 90.0% word accuracy | 1.8MB | Word lookup + Viterbi Bigram LM |
| **Igbo Diacritizer** | ✅ | 95.2% char accuracy | 250KB | Syllable dictionary lookup |
| **Sentiment Analysis** | ✅ | 72% accuracy | 150KB | TF-IDF + Logistic Regression |
| **Tokenizers (Unified)** | ✅ | 100% diacritic preservation | 400KB | BPE Trained on Dedicated Corpora |

---

## Design Philosophy

1. **CPU-first & Lightweight**: Every feature works on 4GB RAM laptops. We use custom Python/NumPy classifiers to avoid pulling PyTorch/TensorFlow.
2. **Offline-capable**: Bundled models are saved in efficient JSON formats directly with the library. 
3. **Honest Metrics**: We share real-world performance benchmarks. We don't hide contextual ambiguity in Yoruba diacritics or classification limitations.

---

## License

Apache License 2.0.
