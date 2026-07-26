# NLP & Tokenization

The `olaverse.nlp` module is the core of the SDK — production-ready tools for African language text processing, detection, diacritization, tokenization, and preprocessing.

```bash
pip install olaverse                # all NLP tools included (no GPU required)
pip install olaverse[deeplearning]  # adds LIDNeural5/LIDNeural25/LIDNeural5_1, diacnet-1.0
pip install olaverse[lid]           # adds LIDLite25 (fastText, 25 languages)
pip install olaverse[retrieval]     # adds Reranker, Embedder
```

---

## Language Detection

Two models cover the same 5 languages at very different scales. Pick based on your latency and accuracy requirements.

<div class="ov-compare-grid">

<div class="ov-compare-card">
  <div class="ov-compare-title">LIDLite5 — Lightweight</div>
  <div class="ov-compare-badge ov-badge-green">Zero GPU · Instant</div>
  <ul>
    <li><strong>1.1 MB</strong> JSON model</li>
    <li><strong>0.014 ms</strong> per sentence</li>
    <li><strong>98.12%</strong> macro accuracy</li>
    <li>TF-IDF + Logistic Regression</li>
    <li>Pure Python — no torch, no transformers</li>
  </ul>
</div>

<div class="ov-compare-card">
  <div class="ov-compare-title">LIDNeural5 — Neural</div>
  <div class="ov-compare-badge ov-badge-purple">GPU Recommended</div>
  <ul>
    <li><strong>484 MB</strong> — XLM-RoBERTa 125M</li>
    <li><strong>13.3 ms</strong> per sentence (CPU/GPU)</li>
    <li><strong>98.96%</strong> macro accuracy</li>
    <li>Fine-tuned on castorini/afriberta_large</li>
    <li>Requires <code>olaverse[deeplearning]</code></li>
  </ul>
</div>

</div>

### LIDLite5

**Model Card**: [olaverse/lid-lite-5](https://huggingface.co/olaverse/lid-lite-5)

```python
from olaverse import LIDLite5, detect_language

# Class interface
detector = LIDLite5()
detector.predict("Bawo ni, se daadaa ni?")      # → 'yor'
detector.predict("Sannu, yaya kake?")            # → 'hau'
detector.predict("How far, wetin dey happen?")   # → 'pcm'

# Probability distribution over all 5 classes
detector.predict_proba("Kedu, ọ dị mma?")
# → {'eng': 0.037, 'hau': 0.024, 'ibo': 0.807, 'pcm': 0.021, 'yor': 0.112}

# Quick one-liner
detect_language("E kaaro, bawo ni?")             # → 'yor'
```

::: olaverse.nlp.LIDLite5
::: olaverse.nlp.detect_language

---

### LIDNeural5

**Model Card**: [olaverse/lid-neural-5](https://huggingface.co/olaverse/lid-neural-5)

```bash
pip install olaverse[deeplearning]
```

```python
from olaverse import LIDNeural5

detector = LIDNeural5()
detector.load()  # downloads once from Hugging Face, cached after first run

detector.predict("Kedu ka ị mere today?")        # → 'ibo'
detector.predict("Ẹ káàárọ̀, ẹ káàbọ̀")        # → 'yor'

probs = detector.predict_proba("How far, wetin dey happen?")
# → {'pcm': 0.991, 'eng': 0.006, 'ibo': 0.001, 'hau': 0.001, 'yor': 0.001}
```

#### Batch Inference

`predict_batch` and `predict_proba_batch` run a **single batched forward pass** — much faster than calling `predict()` in a loop over a dataset.

```python
texts = ["Bawo ni?", "Kedu ọ dị?", "How far?", "Sannu dai."]

# Returns a list of predicted language codes
langs = detector.predict_batch(texts)
# → ['yor', 'ibo', 'pcm', 'hau']

# Returns a list of probability dicts — one per input
probs = detector.predict_proba_batch(["Bawo ni?", "Kedu?"])
# → [
#     {'yor': 0.991, 'eng': 0.005, ...},
#     {'ibo': 0.987, 'eng': 0.008, ...},
#   ]
```

**Per-language accuracy:**

| Language | Precision | Recall | F1-Score |
|---|---|---|---|
| Yoruba (`yor`) | 99.60% | 99.60% | 99.60% |
| Hausa (`hau`) | 99.60% | 99.20% | 99.40% |
| Igbo (`ibo`) | 98.79% | 98.20% | 98.50% |
| Nigerian Pidgin (`pcm`) | 99.20% | 98.80% | 99.00% |
| English (`eng`) | 97.63% | 99.00% | 98.31% |
| **Overall (Macro)** | | | **98.96%** |

::: olaverse.nlp.LIDNeural5

---

### LIDLite25 / LIDNeural25 — 25-language identification

!!! tip "New in v0.1.5"
    Beyond the 5 core Nigerian languages, `LIDLite25` and `LIDNeural25` cover 25 languages spanning Africa, Europe, and Asia — for products that need broader coverage than Yoruba/Hausa/Igbo/Pidgin/English.

**Model Cards**: [olaverse/lid-lite-25](https://huggingface.co/olaverse/lid-lite-25) · [olaverse/lid-neural-25.1](https://huggingface.co/olaverse/lid-neural-25.1) · [olaverse/lid-neural-25.2](https://huggingface.co/olaverse/lid-neural-25.2)

Both come in two checkpoints, tuned for different input lengths — pick the `variant=` that matches your traffic:

| `variant=` | Use for |
|---|---|
| `"passages"` | Documents, articles, paragraph-length text |
| `"questions"` *(default)* | Search queries, chat messages, short user input |

`LIDLite25` is a CPU-only fastText classifier (sub-millisecond, ~5-10MB per checkpoint); `LIDNeural25` is an XLM-RoBERTa-base sequence classifier — higher accuracy on short text (98.2% vs 97.3%), at the cost of needing `transformers`/`torch`.

```bash
pip install olaverse[lid]           # LIDLite25 (fastText)
pip install olaverse[deeplearning]  # LIDNeural25 (transformers)
```

```python
from olaverse import LIDLite25, LIDNeural25

lite = LIDLite25(variant="questions")
lite.predict("What causes ocean tides?")   # → 'eng'

neural = LIDNeural25(variant="questions")
neural.load()
neural.predict_proba("What causes ocean tides?")
# → {'eng': 0.999, 'fra': 0.0003, ...}
```

!!! warning "Zulu/Xhosa confusion on short text"
    Both models score noticeably lower on Zulu/Xhosa short-text classification (F1 ~0.77-0.79) than every other language (≥0.98) — the two languages are closely related with substantial shared vocabulary. This holds across both architectures, so treat predictions between these two specifically with reduced confidence on short input.

::: olaverse.nlp.LIDLite25
::: olaverse.nlp.LIDNeural25

---

### LIDNeural5_1 — Nigerian-only, no English fallback

**Model Card**: [olaverse/lid-neural-5.1](https://huggingface.co/olaverse/lid-neural-5.1)

A compact (~31M parameter) classifier built on [`mist-encoder-base-ng`](https://huggingface.co/olaverse/mist-encoder-base-ng), covering only the 4 main Nigerian languages — no English/"other" class.

```python
from olaverse import LIDNeural5_1

detector = LIDNeural5_1()
detector.predict("Ina kwana?")   # → 'Hausa'
```

!!! danger "No English class"
    Out-of-set input (English or any other language) will be **confidently mislabelled**, most often as Nigerian Pidgin. If your input may include English, use `LIDLite25`/`LIDNeural25`/`LIDNeural5` instead — this model always picks one of the four Nigerian languages.

::: olaverse.nlp.LIDNeural5_1

---

## Diacritization

DiacNet restores tones and diacritical marks stripped from Yoruba and Igbo text — the critical front-end step for TTS, language learning, and NLP accuracy.

### Available Models

| Model ID | Language | Method | Speed | Accuracy | Size |
|---|---|---|---|---|---|
| `diacnet-yor-viterbi` | Yoruba | Viterbi n-gram | ⚡ Fast | Good | ~7 MB |
| `diacnet-yor-db` | Yoruba (dot-below only) | KNN backoff | ⚡ Fast | Dot-below focused | ~2 MB |
| `diacnet-yor` | Yoruba | BiLSTM | Medium | 93.35% char | 2.4 MB |
| `diacnet-yor-x` | Yoruba (full) | XLM-RoBERTa | Slow | 82.46% word | 503 MB |
| `diacnet-ig` | Igbo | KNN backoff | ⚡ Fast | Good | ~3 MB |
| `diacnet-1.0` | 10 languages (see below) | ByT5 seq2seq | Slow | ~0.02 median CER | ~300 MB |

### Quick Functions

```python
from olaverse import diacritize_yoruba, diacritize_yoruba_dot_below, diacritize_igbo

# Yoruba — full tonal diacritics (Viterbi, fast)
diacritize_yoruba("Ojo lo si oja lana")
# → 'Òjó lọ sí ọjà lana'

# Yoruba — dot-below vowels only (KNN)
diacritize_yoruba_dot_below("Ojo lo si oja")
# → 'Ọjọ lo si ọja'

# Igbo
diacritize_igbo("Kedu ka i mere")
# → 'Kedụ ka ị mere'
```

### Unified Diacritizer Class

Use `Diacritizer` when you need to choose a specific model, switch between neural backends, or process mixed-language input.

```python
from olaverse.nlp import Diacritizer

# Default: fast Viterbi for Yoruba
d = Diacritizer(model="diacnet-yor-viterbi")
d.restore("Ojo lo si oja lana")
# → 'Òjó lọ sí ọjà lana'

# High-accuracy neural (requires olaverse[deeplearning])
d_neural = Diacritizer(model="diacnet-yor-x")
d_neural.restore("Ojo lo si oja lana")
```

#### Auto-routing (`model="auto"`) — *New in v0.1.4*

Set `model="auto"` to skip manual language selection. LIDLite5 detects the language on each call and routes to the correct backend — Yoruba → `diacnet-yor-viterbi`, Igbo → `diacnet-ig`. Both LIDLite5 and the target diacritizer are lazy-loaded on first use.

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="auto")

# Yoruba text — routes to diacnet-yor-viterbi
d.restore("Ojo lo si oja lana")
# → 'Òjó lọ sí ọjà lana'

# Igbo text — routes to diacnet-ig
d.restore("Kedu ka i mere")
# → 'Kedụ ka ị mere'
```

#### diacnet-1.0 — multilingual, 10 languages *(New in v0.1.5)*

**Model Card**: [olaverse/diacnet-1.0](https://huggingface.co/olaverse/diacnet-1.0)

A single joint ByT5 model that restores diacritics across 10 languages — Yoruba, Igbo, Hausa, Vietnamese, Polish, Turkish, Portuguese, Spanish, French, and Italian — selected via the `lang=` argument, no separate per-language model or upstream LID step required.

```bash
pip install olaverse[deeplearning]
```

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="diacnet-1.0", lang="fr")
d.restore("Le cafe est tres chaud, mais il prefere le the.")
# → 'Le café est très chaud, mais il préfère le thé.'

d_yo = Diacritizer(model="diacnet-1.0", lang="yo")
d_yo.restore("se eranko naa si gbo o?")
# → 'ṣé ẹranko náà sì gbọ́ ọ?'
```

Supported `lang=` codes: `"yo", "vi", "ig", "ha", "pl", "tr", "pt", "es", "fr", "it"`.

!!! warning "Feed it complete sentences — short fragments degenerate"
    `diacnet-1.0` is a seq2seq model, not a per-character tagger, so it can
    rewrite the text rather than only adding marks. On full sentences this is
    reliable. On short fragments it fails in several ways:

    | Input | `lang=` | Output |
    |---|---|---|
    | `"el nino"` | `es` | `'el niño\nel niño\nel niño…'` (repetition loop) |
    | `"je nai pas"` | `fr` | `'je nai pas pas je nai pas pas…'` (repetition loop) |
    | `"cafe"` | `fr` | `'cafẹ́'` (Yoruba dot-below on a French word) |
    | `"nino"` | `es` | `'niños'` (inflection changed) |
    | `"pho"` | `vi` | `'phối>'` (wrong word + stray character) |
    | `"citta"` | `it` | `'città?'` (punctuation invented) |

    Pass whole sentences with their punctuation. It also restores **diacritics
    only** — it does not insert apostrophes, so `"cest fini"` does not become
    `"c'est fini"`.

!!! warning "Yoruba is the hardest language for this model"
    Yoruba's median CER (0.110) is nearly 3x the next-highest language — genuine tonal ambiguity (the same base letters can carry multiple valid tone patterns), not a model weakness. For Yoruba specifically, the dedicated `diacnet-yor-viterbi`/`diacnet-yor-x` models above may perform better; `diacnet-1.0`'s advantage is breadth (10 languages, one model), not peak Yoruba accuracy.

::: olaverse.nlp.Diacritizer
::: olaverse.nlp.diacritize_yoruba
::: olaverse.nlp.diacritize_yoruba_dot_below
::: olaverse.nlp.diacritize_igbo

---

## Tokenization — OTK-BPE-50k

**Model Card**: [olaverse/otk-bpe-50k](https://huggingface.co/olaverse/otk-bpe-50k)

Byte-Level BPE tokenizers trained on dedicated corpora for each Nigerian language. **0% out-of-vocabulary** tokens via raw UTF-8 byte fallback.

### Why OTK-BPE?

General-purpose tokenizers like GPT-4's cl100k tokenize African languages character-by-character, blowing up sequence lengths and wasting context. OTK-BPE-50k learns proper subwords from native text:

| Model | Language | Vocab | Efficiency vs GPT-4 |
|---|---|---|---|
| `otk-bpe-50k-yo` | Yoruba | 50,000 | **63% fewer tokens** |
| `otk-bpe-50k-ig` | Igbo | 50,000 | ~60% fewer tokens |
| `otk-bpe-50k-ha` | Hausa | 50,000 | ~58% fewer tokens |
| `otk-bpe-50k-pcm` | Nigerian Pidgin | 50,000 | ~55% fewer tokens |
| `otk-bpe-50k-naija` | Unified (all 4) | 50,000 | Balanced |

### Usage

```python
from olaverse import Tokenizer

# Load by language code
tok_yo = Tokenizer("yo")      # Yoruba
tok_ig = Tokenizer("ig")      # Igbo
tok_ha = Tokenizer("ha")      # Hausa
tok_pcm = Tokenizer("pcm")    # Nigerian Pidgin
tok_all = Tokenizer("naija")  # Unified

# Encode / decode
ids = tok_yo.encode("Ẹ kú àbọ̀")
print(ids)                      # → [124, 381]
print(tok_yo.decode(ids))       # → 'Ẹ kú àbọ̀'

# For fine-tuning / dataset preparation
sentences = ["Bawo ni?", "Se daadaa ni?", "Mo dupe."]
all_ids = [tok_yo.encode(s) for s in sentences]
```

::: olaverse.nlp.Tokenizer

---

## Tokenization — OTK-BPE (Multilingual) *(New in v0.1.5)*

**Model Card**: [olaverse/otk-bpe](https://huggingface.co/olaverse/otk-bpe)

A companion tokenizer family for **Swahili**, **Kinyarwanda**, and a **merged French + Kinyarwanda + English + Swahili** vocabulary — each available at three vocab sizes. Same `Tokenizer` class, different `lang=` values.

| `lang=` | Languages | Vocab sizes |
|---|---|---|
| `sw-50k` / `sw-100k` / `sw-150k` | Swahili | 50k / 100k / 150k |
| `kin-50k` / `kin-100k` / `kin-150k` | Kinyarwanda | 50k / 100k / 150k |
| `merged-50k` / `merged-100k` / `merged-150k` | French + Kinyarwanda + English + Swahili | 50k / 100k / 150k |

**150k is the recommended default** — fertility and entity-handling both improve monotonically from 50k → 100k → 150k in every benchmark on the model card, with no exceptions. Step down only if embedding-table size is a hard constraint.

```python
from olaverse import Tokenizer

tok = Tokenizer("sw-150k")
ids = tok.encode("Habari yako? Leo ni siku nzuri sana 😊")
tok.decode(ids)
# → 'Habari yako? Leo ni siku nzuri sana 😊'

tok_merged = Tokenizer("merged-150k")  # French + Kinyarwanda + English + Swahili
```

---

## Text Normalization (TTS)

`TTSNormalizer` converts raw text into a form suitable for speech synthesis — expanding numbers, abbreviations, and punctuation into their spoken equivalents.

```python
from olaverse import TTSNormalizer

# Yoruba normalization
norm = TTSNormalizer(lang="yo")
norm.normalize("Dr. Ade lo si oja lana")
# → 'Dọ́kítà Ade lo si oja lana'

norm.normalize("O san ₦1,200")
# → 'O san  ọ̀kan ẹgbẹ̀rún méjì'

# Igbo normalization
norm_ig = TTSNormalizer(lang="ig")
norm_ig.normalize("Prof. Obi ra ulo")
# → 'Purofesọ Obi ra ulo'
```

::: olaverse.nlp.TTSNormalizer

---

## Nigerian Pidgin Normalization — NaijaNormalizer

**`NaijaNormalizer`** is a Pidgin-specific TTS normalizer built on top of `TTSNormalizer`. It adds a pre-processing step that expands informal Pidgin spellings — SMS shorthand, phonetic spellings, loan abbreviations — before running the standard abbreviation and number expansion.

```python
from olaverse import NaijaNormalizer

norm = NaijaNormalizer()

# Informal spellings expanded first, then standard normalization
norm.normalize("Oga, 2moro na Sunday. Call am nd tell am 2 come.")
# → 'Oga, tomorrow na Sunday. Call am and tell am to come.'

norm.normalize("e don finish. tnx 4 d help, u r d best!")
# → 'e don finish. thanks for the help, you are the best!'

# normalize_informal() only — skip abbreviations/number expansion
norm.normalize_informal("dis tin dey hard nd i no sabi wetin 2 do")
# → 'this tin dey hard and i no sabi wetin to do'
```

**Informal expansion map includes** (35+ entries):

| Input | Output | Input | Output |
|---|---|---|---|
| `2moro` | tomorrow | `2day` | today |
| `b4` | before | `4` | for |
| `dis` | this | `dat` | that |
| `nd` / `n` | and | `u` | you |
| `pls` | please | `tnx` | thanks |
| `lol` | laugh | `smh` | sigh |
| `jst` | just | `wat` | what |
| `wen` | when | `hw` | how |

::: olaverse.nlp.NaijaNormalizer

---

## Stopwords

The stopwords module provides curated stopword sets for all 4 Nigerian languages, ready for use in NLP pipelines, TF-IDF vectorizers, and preprocessing workflows.

```python
from olaverse import (
    YORUBA_STOPWORDS, IGBO_STOPWORDS, HAUSA_STOPWORDS, PIDGIN_STOPWORDS,
    get_stopwords, filter_stopwords,
)
```

### Available Sets

| Constant | Language | Example words |
|---|---|---|
| `YORUBA_STOPWORDS` | Yoruba | mo, mi, o, ni, ní, fun, lati, ati, tabi, sugbon, ṣugbọn |
| `IGBO_STOPWORDS` | Igbo | m, mu, gi, gị, ya, anyi, anyị, ka, na, nke |
| `HAUSA_STOPWORDS` | Hausa | ni, kai, ke, shi, ita, mu, ku, su, da, ko, amma |
| `PIDGIN_STOPWORDS` | Nigerian Pidgin | i, you, him, na, be, go, don, dey, de, wey, sey |

All sets are `frozenset` — safe to use in `in` membership tests and `set` operations.

### Usage

```python
# Direct membership test
"ni" in YORUBA_STOPWORDS   # → True
"atal" in IGBO_STOPWORDS   # → False

# Retrieve by language code
get_stopwords("yor")   # → YORUBA_STOPWORDS  (also accepts "yo")
get_stopwords("ibo")   # → IGBO_STOPWORDS    (also accepts "ig")
get_stopwords("hau")   # → HAUSA_STOPWORDS   (also accepts "ha")
get_stopwords("pcm")   # → PIDGIN_STOPWORDS
get_stopwords("eng")   # → frozenset of common English stopwords

# Filter a tokenized sentence
tokens = ["Bawo", "ni", "Ade", "ati", "Sade", "dara"]
filter_stopwords(tokens, "yor")
# → ['Bawo', 'Ade', 'Sade', 'dara']

# Works with any iterable
filter_stopwords(iter(["dem", "dey", "Lagos", "for", "here"]), "pcm")
# → ['Lagos']
```

### Using with scikit-learn

```python
from olaverse import get_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

yoruba_docs = ["Bawo ni Ade?", "Sade dara pupo", "Mo feran onje Yoruba"]
stop = list(get_stopwords("yor"))

vec = TfidfVectorizer(stop_words=stop)
X = vec.fit_transform(yoruba_docs)
```

---

## Text Preprocessing & Cleaning

### PII Masking

```python
from olaverse import mask_pii

mask_pii("Contact me at support@olaverse.co.uk or call +1-800-555-0199")
# → 'Contact me at [EMAIL] or call [PHONE]'

mask_pii("My card is 4111-1111-1111-1111 and SSN 123-45-6789")
# → 'My card is [CREDIT_CARD] and SSN [SSN]'
```

::: olaverse.nlp.mask_pii

---

### Text Cleaning

```python
from olaverse import clean_text

clean_text("Visit <a href='https://olaverse.co.uk'>our site</a>  today!")
# → 'Visit our site today!'

clean_text("Check https://example.com for details.  Multiple   spaces.")
# → 'Check  for details. Multiple spaces.'

# Keep URLs
clean_text("Visit https://olaverse.co.uk", remove_urls=False)
# → 'Visit https://olaverse.co.uk'
```

::: olaverse.nlp.clean_text

---

## Retrieval *(New in v0.1.5)*

```bash
pip install olaverse[retrieval]
```

Two-piece toolkit for building RAG/search pipelines: a cross-encoder `Reranker` for the second stage, and a Nigerian-language `Embedder` for semantic search and cross-lingual retrieval.

### Reranker

**Model Cards**: [olaverse/mist-reranker-150m](https://huggingface.co/olaverse/mist-reranker-150m) · [olaverse/mist-reranker-22.7M](https://huggingface.co/olaverse/mist-reranker-22.7M)

Scores `(query, passage)` pairs to re-sort the top-k candidates from a first-stage retriever (BM25 or a bi-encoder).

| `size=` | Model | Params | Best for |
|---|---|---|---|
| `"150m"` | mist-reranker-150m | ~150M | Best QA/fact accuracy (ModernBERT-base) |
| `"22.7m"` *(default)* | mist-reranker-22.7M | ~22.7M | Smaller/faster, MiniLM-L6 backbone |

```python
from olaverse import Reranker

reranker = Reranker(size="22.7m")

reranker.rank("who wrote hamlet", [
    "Hamlet is a tragedy written by William Shakespeare.",
    "The capital of France is Paris.",
])
# → [(0, 0.915...), (1, 0.301...)]   # (original_index, score), best-first

reranker.score("who wrote hamlet", ["Hamlet is a tragedy by Shakespeare."])
# → [0.912...]
```

Both models are English-only; `Reranker` auto-handles their different output head shapes (a single relevance score vs. 2-class logits).

::: olaverse.nlp.Reranker

---

### Embedder

**Model Card**: [olaverse/naija-embed-base](https://huggingface.co/olaverse/naija-embed-base)

Cross-lingual sentence embeddings for Hausa, Yoruba, and Igbo — contrastively fine-tuned from [`mist-encoder-base-ng`](https://huggingface.co/olaverse/mist-encoder-base-ng). Useful for cross-lingual retrieval (e.g. Hausa query → Yoruba document), semantic search, clustering, and deduplication.

```python
from olaverse import Embedder

embedder = Embedder()
vecs = embedder.encode(["bawo ni", "sannu"])
embedder.similarity(vecs[0], vecs[1])
```

!!! note "No Nigerian Pidgin support"
    The underlying translation model used for training only outputs Hausa/Yoruba/Igbo — Pidgin (`pcm`) is not covered.

::: olaverse.nlp.Embedder
