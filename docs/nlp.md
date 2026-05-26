# NLP & Tokenization

The `olaverse.nlp` module provides tools for text manipulation, optimized tokenization, speech-ready normalization, diacritization for African languages, and PII masking.

---

## Text Preprocessing & Cleaning

Utilities to strip HTML markup, clean up whitespace, and remove URLs from raw text.

```python
from olaverse.nlp import clean_text

clean_text("Read more at <a href='https://olaverse.ai'>Olaverse</a>!   ")
# → 'Read more at Olaverse!'
```

::: olaverse.nlp.clean_text

---

## Global PII Masking

Securely redacts sensitive personal identifiers — emails, global phone numbers, credit card numbers, and social security numbers — from any text.

```python
from olaverse.nlp import mask_pii

mask_pii("Contact me at support@olaverse.co.uk or call +1-800-555-0199")
# → 'Contact me at [EMAIL] or call [PHONE]'

mask_pii("My card is 4111-1111-1111-1111 and SSN 123-45-6789")
# → 'My card is [CREDIT_CARD] and SSN [SSN]'
```

::: olaverse.nlp.mask_pii

---

## Text Normalization (TTS)

`TTSNormalizer` converts raw text into a form suitable for speech synthesis — expanding numbers, currencies, abbreviations, and date formats into their spoken equivalents.

```python
from olaverse.nlp import TTSNormalizer

norm = TTSNormalizer()
print(norm.normalize("She paid $1,200.50 on 12/25/2023"))
# → 'She paid one thousand two hundred dollars and fifty cents on December twenty-fifth twenty twenty-three'
```

::: olaverse.nlp.TTSNormalizer

---

## Language Detection

### LIDLite5 — Lightweight (Zero-dependency)

**Model Card**: [olaverse/lid-lite-5](https://huggingface.co/olaverse/lid-lite-5)

`LIDLite5` uses a custom TF-IDF (unigram + bigram) feature extractor with a Logistic Regression classifier. The entire model is a single ~1.1 MB JSON file. No deep learning required.

| Property | Value |
|---|---|
| Model Type | TF-IDF + Logistic Regression |
| Vocabulary | 5,000 top n-gram terms |
| File Size | 1.10 MB |
| Accuracy | **98.12%** (Macro) |
| Latency | **0.014 ms/sentence** |
| Dependencies | Pure Python + NumPy |

```bash
pip install olaverse  # no extra dependencies needed
```

```python
from olaverse import LIDLite5

detector = LIDLite5()

# Predict dominant language
lang = detector.predict("Bawo ni, se daadaa ni?")
print(lang)  # → 'yor'

# Get confidence scores across all 5 classes
probs = detector.predict_proba("How far, wetin dey happen?")
print(probs)
# → {'eng': 0.006, 'hau': 0.001, 'ibo': 0.002, 'pcm': 0.989, 'yor': 0.002}
```

### detect_language — Quick one-liner

```python
from olaverse.nlp import detect_language

detect_language("E kaaro, bawo ni?")  # → 'yor'
```

::: olaverse.nlp.detect_language
::: olaverse.nlp.LIDLite5

---

## Diacritization

### Yoruba Diacritizer

**Model Card**: [olaverse/diacnet-yor](https://huggingface.co/olaverse/diacnet-yor) · [olaverse/diacnet-yor-x](https://huggingface.co/olaverse/diacnet-yor-x)

Two complementary Yoruba diacritization models are available:

| Model | Architecture | Word Accuracy | Character Accuracy | Size |
|---|---|---|---|---|
| `diacnet-yor` (dot-below only) | BiLSTM | — | 93.35% char | 2.4 MB |
| `diacnet-yor-x` (full tonal + dot-below) | AfriBERTa Transformer | 82.46% | — | 503 MB |

The `Diacritizer` class uses a **Viterbi decoding** approach on top of the transformer model for full tonal restoration, achieving **90.0% word accuracy** on evaluation sets.

```python
from olaverse.nlp import diacritize_yoruba, diacritize_yoruba_dot_below

# Full tonal diacritics (uses diacnet-yor-x with Viterbi decoding)
diacritize_yoruba("Ojo lo si oja lana")
# → 'Ọjọ́ ló sí ọjà lànà'

# Dot-below only (fast, uses diacnet-yor BiLSTM)
diacritize_yoruba_dot_below("Ojo lo si oja")
# → 'Ọjọ lo si ọja'
```

### Igbo Diacritizer

```python
from olaverse.nlp import diacritize_igbo

diacritize_igbo("Kedu ka i mere")
# → 'Kedụ ka ị mere'
```

::: olaverse.nlp.Diacritizer
::: olaverse.nlp.diacritize_yoruba
::: olaverse.nlp.diacritize_igbo

---

## Tokenization — OTK-BPE-50k

**Model Card**: [olaverse/otk-bpe-50k](https://huggingface.co/olaverse/otk-bpe-50k)

`Tokenizer` wraps the **OTK-BPE-50k** family of Byte-Level BPE tokenizers, each trained on dedicated African language corpora. They achieve **0% out-of-vocabulary (OOV)** tokens via raw UTF-8 byte mapping.

| Model | Language | Vocab | vs GPT-4 |
|---|---|---|---|
| `otk-bpe-50k-yo` | Yoruba | 50k | 63% fewer tokens |
| `otk-bpe-50k-ig` | Igbo | 50k | ~60% fewer tokens |
| `otk-bpe-50k-ha` | Hausa | 50k | ~58% fewer tokens |
| `otk-bpe-50k-pcm` | Nigerian Pidgin | 50k | ~55% fewer tokens |
| `otk-bpe-50k-naija` | Unified (all 4) | 50k | Balanced |

```bash
pip install olaverse  # tokenizer library included
```

```python
from olaverse.nlp import Tokenizer

# Load by short code: "yo", "ig", "ha", "pcm", or "naija"
tok = Tokenizer("yo")

tokens = tok.encode("Ẹ kú àbọ̀")
print(tokens)   # → [124, 381]

decoded = tok.decode(tokens)
print(decoded)  # → 'Ẹ kú àbọ̀'
```

::: olaverse.nlp.Tokenizer
