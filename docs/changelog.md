# Changelog

All notable changes to the Olaverse SDK are documented here.

---

## v0.2.1 — *Current*

**Released**: 2026-07-26

Documentation only — no code changes. The README shipped in v0.2.0 predated the new models, so the PyPI project page didn't mention `MISTTitleGenerator` or `MISTQuestionGenerator`. PyPI renders the README from the uploaded distribution and won't let a published version be edited, so refreshing that page needs a release.

---

## v0.2.0

**Released**: 2026-07-26

### New Features

#### `MISTTitleGenerator` and `MISTQuestionGenerator`

The two task-specific MIST models now have SDK classes instead of requiring raw `transformers` code.

```python
from olaverse import MISTTitleGenerator, MISTQuestionGenerator

MISTTitleGenerator().generate("My laptop keeps freezing when I open too many tabs, why?")
# → 'Laptop Freezing Impact'

MISTQuestionGenerator().generate(passage, n=3, language="fra")
```

`MISTQuestionGenerator` uses the teacher prompt the model was distilled with, sizes the JSON skeleton to `n`, and accepts ISO 639-3, ISO 639-1, or English language names. `language=` names the language the passage is written in — this is same-language question generation, not translation.

### Fixes

#### `diacnet-1.0` no longer truncates long text

The model was trained on sentence-length input (median 58 bytes) and the SDK fed whole paragraphs through in one pass, silently losing data — a 358-character French paragraph came back at 235 characters with its last two sentences dropped. Multi-sentence input is now segmented, restored a sentence at a time, and rejoined.

```python
Diacritizer(model="diacnet-1.0", lang="fr")                         # segments (default)
Diacritizer(model="diacnet-1.0", lang="fr", split_sentences=False)  # one pass
Diacritizer(model="diacnet-1.0", lang="fr", splitter=my_splitter)   # your own
```

### Documentation

Model claims were re-run against the real checkpoints and corrected where they didn't hold: `PrismDenoiser` always returns 128x128 rather than matching input, `PrismSteganography` messages do not survive a JPEG save at any quality, and the multilingual OTK-BPE tokenizers silently drop emoji.

---

## v0.1.5

**Released**: 2026-07-16

### New Features

#### 25-language identification (`LIDLite25`, `LIDNeural25`)

Extends language detection well beyond the original 5 Nigerian languages to 25 languages spanning Africa, Europe, and Asia. Each comes in `"passages"` (long-form) and `"questions"` (short-form) variants.

```python
from olaverse import LIDLite25, LIDNeural25

lite = LIDLite25(variant="questions")       # fastText, CPU-only
lite.predict("What causes ocean tides?")    # → 'eng'

neural = LIDNeural25(variant="questions")   # XLM-RoBERTa
neural.load()
neural.predict_proba("What causes ocean tides?")
```

#### `LIDNeural5_1` — compact Nigerian-only classifier

A ~31M parameter classifier built on the new `mist-encoder-base-ng` encoder, covering Hausa/Yoruba/Igbo/Nigerian Pidgin with no English fallback class.

```python
from olaverse import LIDNeural5_1

detector = LIDNeural5_1()
detector.predict("Ina kwana?")   # → 'Hausa'
```

#### `diacnet-1.0` — multilingual diacritization

A single joint ByT5 model restores diacritics across 10 languages (Yoruba, Igbo, Hausa, Vietnamese, Polish, Turkish, Portuguese, Spanish, French, Italian) via a `lang=` argument — no separate per-language model needed.

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="diacnet-1.0", lang="fr")
d.restore("Le cafe est tres chaud.")   # → 'Le café est très chaud.'
```

#### OTK-BPE multilingual tokenizer family

Swahili, Kinyarwanda, and a merged French/Kinyarwanda/English/Swahili tokenizer, each at 50k/100k/150k vocab sizes, through the same `Tokenizer` class used for the Nigerian-language family.

```python
from olaverse import Tokenizer

tok = Tokenizer("sw-150k")
ids = tok.encode("Habari yako?")
```

#### New retrieval toolkit (`olaverse.nlp.retrieval`)

`Reranker` (cross-encoder, 150M/22.7M variants) and `Embedder` (cross-lingual Hausa/Yoruba/Igbo sentence embeddings) for RAG/search pipelines.

```python
from olaverse import Reranker, Embedder

reranker = Reranker(size="22.7m")
reranker.rank("who wrote hamlet", ["Hamlet is a tragedy by Shakespeare.", "Paris is in France."])

embedder = Embedder()
vecs = embedder.encode(["bawo ni", "sannu"])
```

#### Dataset access (`olaverse.data`)

`load_dataset`, `list_datasets`, and `dataset_info` give one-line access to every public olaverse dataset on Hugging Face — reranker training pairs, multilingual QG passages, and the DiacBench diacritization benchmark. Requires `pip install olaverse[data]`. Replaces the old placeholder `olaverse.data.loaders` module, which returned bundled sample rows instead of real data.

```python
from olaverse import load_dataset

pairs = load_dataset("reranker-general-en-llm-judged", split="train")
bench = load_dataset("diacbench", "yo", split="test")
```

#### New `olaverse.vision` module

`PrismUpscaler` (2x/4x/arbitrary-resolution super-resolution), `PrismDenoiser` (noise/blur/JPEG-artifact removal), and `PrismSteganography` (hide/recover short messages in images) — general-purpose image-to-image models, not African-language-specific.

```python
from olaverse import PrismUpscaler, PrismDenoiser, PrismSteganography

PrismUpscaler(size="2x").upscale("input.jpg").save("output.jpg")
PrismDenoiser().denoise("noisy.jpg").save("denoised.jpg")

steg = PrismSteganography()
stego = steg.hide("cover.jpg", "hi there")
steg.reveal(stego)   # → 'hi there'
```

### Changes

- **New extras**: `olaverse[lid]` (fastText for `LIDLite25`), `olaverse[retrieval]` (`sentence-transformers`), `olaverse[vision]` (`torch`, `torchvision`, `Pillow`), `olaverse[data]` (Hugging Face `datasets`).
- **Fix — `diacritize_yoruba`** previously special-cased the documentation example sentence and returned a hand-written answer for it instead of the model output. The special case is removed; all inputs now go through the Viterbi model, and docs show the model's real output.
- **`Diacritizer`** gained a `lang=` constructor argument, used only by `model="diacnet-1.0"`.
- **`Tokenizer`** now resolves multilingual (`sw-*`/`kin-*`/`merged-*`) variants against the `olaverse/otk-bpe` repo, alongside the existing Nigerian-language `olaverse/otk-bpe-50k` repo.
- **`LIDNeural5`** internals refactored onto a shared base class (`LIDNeural25`, `LIDNeural5_1` reuse the same loading/inference logic) — no change to `LIDNeural5`'s public API.

### Install

```bash
pip install olaverse                # core NLP (no GPU required)
pip install olaverse[deeplearning]  # + LIDNeural5/25/51, diacnet-1.0, MIST local
pip install olaverse[lid]           # + LIDLite25 (fastText)
pip install olaverse[retrieval]     # + Reranker, Embedder
pip install olaverse[vision]        # + PrismUpscaler, PrismDenoiser, PrismSteganography
pip install olaverse[hosted]        # + MIST via Featherless/Modal
pip install olaverse[legal]         # + LegalPeace
pip install olaverse[data]          # + load_dataset (olaverse datasets on HF)
```

---

## v0.1.4

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
d.restore("Ojo lo si oja lana")   # detected: Yoruba → 'Òjó lọ sí ọjà lana'
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
    - **`LIDNeural5_1` v5.2** — adds an English/"other" class, removing the confident-mislabelling failure mode of v5.1
    - **`MIST` embedding endpoint** for semantic search over Nigerian language content
    - **ASR (Automatic Speech Recognition)** for Nigerian languages
    - **Hausa / Pidgin TTS normalizer expansion**

!!! success "Done — was on the roadmap"
    - ~~`LIDNeural5` expanded to 10+ languages~~ — shipped in v0.1.5 as `LIDLite25`/`LIDNeural25` (25 languages total: Afrikaans, Amharic, German, English, French, Hausa, Hindi, Igbo, Indonesian, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Russian, Shona, Somali, Spanish, Swahili, Turkish, Vietnamese, Xhosa, Yoruba, Zulu). Note: Efik, Tiv, and Nupe specifically are still not covered by any olaverse LID model.
