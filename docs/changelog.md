# Changelog

All notable changes to the Olaverse SDK are documented here.

---

## v0.1.4 — *Current*

**Released**: 2026-06-15

### New Features

#### MIST Model Family (`olaverse.llm.MIST`)

Unified interface for all MIST variants — correct stop tokens, verified sampling defaults, and a local/hosted endpoint switch in one class.

```python
from olaverse import MIST

# Local (transformers)
model = MIST(size="8b")
model.load()
print(model.generate("What makes Yoruba a tonal language?"))

# Hosted (Featherless, Modal, or any OpenAI-compatible endpoint)
model = MIST(size="70b", endpoint="featherless", api_key="...")
print(model.generate("Write a Python retry decorator."))
```

Supported variants: `"8b"` / `"mini"`, `"70b"`, `"140b"`, `"140b-4bit"`, `"thinking"`.

#### Batch inference — `LIDNeural5.predict_batch` / `predict_proba_batch`

Single batched forward pass instead of per-text loops — significantly faster for dataset processing.

```python
detector = LIDNeural5()
detector.load()

langs = detector.predict_batch(["Bawo ni?", "Kedu?", "How far?"])
# → ['yor', 'ibo', 'pcm']

probs = detector.predict_proba_batch(["Bawo ni?", "Kedu?"])
# → [{'yor': 0.99, ...}, {'ibo': 0.98, ...}]
```

#### Auto-routing Diacritizer (`model="auto"`)

Detects language automatically via LIDLite5 and routes to the correct diacritizer — no need to specify the language.

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="auto")
d.restore("Ojo lo si oja lana")   # detected: Yoruba → 'Ọjọ́ ló sí ọjà lànà'
d.restore("Kedu ka i mere")       # detected: Igbo   → 'Kedụ ka ị mere'
```

#### Stopwords (`olaverse.nlp.stopwords`)

Linguistic stopword sets for all 4 Nigerian languages plus convenience utilities.

```python
from olaverse import YORUBA_STOPWORDS, get_stopwords, filter_stopwords

# Direct set access
"ni" in YORUBA_STOPWORDS        # → True

# By language code
sw = get_stopwords("pcm")       # → PIDGIN_STOPWORDS

# Filter a token list
filter_stopwords(["bawo", "ni", "Ade", "dara"], "yor")
# → ['Ade', 'dara']
```

#### NaijaNormalizer (`olaverse.nlp.NaijaNormalizer`)

Pidgin-specific TTS normalizer extending `TTSNormalizer`. Adds informal spelling normalization (e.g. `"2moro"` → `"tomorrow"`, `"nd"` → `"and"`) on top of the standard abbreviation + number pipeline.

```python
from olaverse import NaijaNormalizer

norm = NaijaNormalizer()
norm.normalize("Oga, e don finish. Call am 2moro pls.")
# → 'Oga, e don finish. Call am tomorrow please.'
```

#### `MIST` retry logic for hosted inference

Automatic retry on capacity/overload errors with configurable attempts and delay.

```python
model = MIST(
    size="70b",
    endpoint="featherless",
    api_key="...",
    max_retries=3,      # default: 3
    retry_delay=5.0,    # seconds; each attempt waits delay × attempt_number
)
```

### Changes

- **`LIDNeural5` moved to `olaverse.nlp`** — its correct home alongside `LIDLite5`. `from olaverse.llm import LIDNeural5` continues to work (backward-compat re-export in `llm/detector.py`).
- **`LIDNeural5` now exported from `olaverse.nlp`** — `from olaverse.nlp import LIDNeural5` is now the canonical import path.
- **Speech demoted to Experimental** — `TTSPipeline`, `BaseAcousticModel`, `BaseVocoder` emit `ExperimentalWarning` on use. No trained acoustic model or vocoder exists yet. The diacritization and normalization steps remain production-ready.
- **`NaijaNormalizer` added to `TTSNormalizer` Pidgin support** — `TTSNormalizer(lang="pcm")` now has a populated abbreviation and digit table (was previously empty).
- **New `olaverse[hosted]` extra** — `pip install olaverse[hosted]` installs `openai>=1.0.0` for MIST hosted inference.
- **`ExperimentalWarning`** exported from `olaverse` and `olaverse.speech` for easy suppression.

### Install

```bash
pip install olaverse           # core NLP (no GPU required)
pip install olaverse[deeplearning]  # + LIDNeural5, MIST local
pip install olaverse[hosted]        # + MIST via Featherless/Modal
pip install olaverse[legal]         # + LegalPeace
```

---

## v0.1.3

**Released**: 2026-05-01 *(approximate)*

### Features

- **`LegalPeace`** — Fine-tuned Mistral-7B-v0.3 for contract analysis and legal reasoning. 4-bit quantized inference via unsloth. Achieves 10.3% faster inference and 32.6% faster contract analysis vs. base Mistral-7B.
- **`LIDNeural5`** *(initially in `olaverse.llm`)* — XLM-RoBERTa sequence classifier fine-tuned on 5 Nigerian languages, 98.96% macro-F1. Available via `pip install olaverse[deeplearning]`.
- **`LIDLite5`** — TF-IDF + Logistic Regression language detector. Zero GPU, 1.1 MB model file, 0.014 ms/sentence, 98.12% macro-F1. Available in core install.
- **`Diacritizer`** with 5 backends — Viterbi, KNN, dot-below KNN, BiLSTM, and XLM-RoBERTa transformer for Yoruba; KNN for Igbo.
- **`Tokenizer`** — OTK-BPE-50k family: Yoruba (63% fewer tokens vs GPT-4), Igbo, Hausa, Pidgin, and unified Naija.
- **`TTSNormalizer`** — Abbreviation and number expansion for Yoruba and Igbo TTS.
- **`mask_pii`** / **`clean_text`** — PII masking (emails, phones, credit cards, SSNs) and general text cleaning.
- **`TTSPipeline`** + **`BaseAcousticModel`** + **`BaseVocoder`** — TTS pipeline architecture (NLP front-end only; acoustic synthesis in development).
- **`olaverse.utils`** — Nigerian currency formatting, continent codes, `.wav` audio I/O.

---

## Roadmap

!!! note "Coming in future releases"
    - **Acoustic model + vocoder** for end-to-end Yoruba TTS (completes the speech pipeline)
    - **`Diacritizer` for Hausa** (`diacnet-ha`)
    - **`LIDNeural5` expanded to 10+ languages** (including Efik, Tiv, Nupe)
    - **`MIST` embedding endpoint** for semantic search over Nigerian language content
    - **ASR (Automatic Speech Recognition)** for Nigerian languages
    - **Hausa / Pidgin TTS normalizer expansion**
