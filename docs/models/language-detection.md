# Language Detection (LID)

**Seven language-identification models, from a 1.1 MB pure-Python classifier to 608-language neural models — pick by coverage, latency, and accuracy.**

```python
from olaverse import detect_language

detect_language("Bawo ni, se daadaa ni?")   # → 'yor'
```

---

## Who is this for?

- Routing user messages to the right model, translator, or support queue
- Filtering or labelling multilingual corpora at scale
- The first stage of pipelines (e.g. `Diacritizer(model="auto")` uses `LIDLite5` internally)

---

## Which model should I use?

| Need | Model | Install |
|---|---|---|
| Nigerian languages + English, zero GPU, instant | `LIDLite5` | `olaverse` |
| Nigerian languages + English, best accuracy | `LIDNeural5` | `olaverse[deeplearning]` |
| 25 languages, CPU-only, sub-millisecond | `LIDLite25` | `olaverse[lid]` |
| 25 languages, best short-text accuracy | `LIDNeural25` | `olaverse[deeplearning]` |
| Only the 4 Nigerian languages — input never contains English | `LIDNeural5_1` | `olaverse[deeplearning]` |
| 608 languages, CPU-only, fast bulk filtering | `LIDLite608` | `olaverse[lid]` |
| 608 languages, best accuracy, especially short text | `LIDNeural608` | `olaverse[deeplearning]` |

---

## LIDLite5 vs LIDNeural5

Both cover Yoruba, Igbo, Hausa, Nigerian Pidgin, and English.

| | LIDLite5 | LIDNeural5 |
|---|---|---|
| Size | **1.1 MB** JSON | 484 MB (XLM-RoBERTa 125M) |
| Speed | **0.014 ms**/sentence | 13.3 ms/sentence |
| Macro accuracy | 98.12% | **98.96%** |
| Architecture | TF-IDF + Logistic Regression | Fine-tuned afriberta_large |
| Dependencies | Pure Python | `transformers` + `torch` |

**Model Cards**: [olaverse/lid-lite-5](https://huggingface.co/olaverse/lid-lite-5) · [olaverse/lid-neural-5](https://huggingface.co/olaverse/lid-neural-5)

```python
from olaverse import LIDLite5, LIDNeural5

lite = LIDLite5()
lite.predict("Sannu, yaya kake?")            # → 'hau'
lite.predict_proba("Kedu, ọ dị mma?")
# → {'ibo': 0.807, 'yor': 0.112, 'eng': 0.037, 'hau': 0.024, ...}

neural = LIDNeural5()
neural.load()   # one-time download, cached
neural.predict("Kedu ka ị mere today?")      # → 'ibo'

# Batched — single forward pass, much faster on datasets
neural.predict_batch(["Bawo ni?", "Kedu ọ dị?", "How far?", "Sannu dai."])
# → ['yor', 'ibo', 'pcm', 'hau']
```

---

## LIDLite25 / LIDNeural25 — 25 languages

Coverage across Africa, Europe, and Asia. Both come in two checkpoints tuned for input length — pick `variant=` to match your traffic:

| `variant=` | Use for |
|---|---|
| `"passages"` | Documents, articles, paragraph-length text |
| `"questions"` *(default)* | Search queries, chat messages, short user input |

**Model Cards**: [olaverse/lid-lite-25](https://huggingface.co/olaverse/lid-lite-25) · [olaverse/lid-neural-25.1](https://huggingface.co/olaverse/lid-neural-25.1) · [olaverse/lid-neural-25.2](https://huggingface.co/olaverse/lid-neural-25.2)

```python
from olaverse import LIDLite25, LIDNeural25

lite = LIDLite25(variant="questions")    # fastText, CPU, ~5-10 MB
lite.predict("What causes ocean tides?")   # → 'eng'

neural = LIDNeural25(variant="questions")  # XLM-RoBERTa-base
neural.load()
neural.predict_proba("What causes ocean tides?")
# → {'eng': 0.999, 'fra': 0.0003, ...}
```

`LIDNeural25` is more accurate on short text (98.2% vs 97.3%) at the cost of needing `transformers`/`torch`.

!!! warning "Zulu/Xhosa confusion on short text"
    Both models score noticeably lower on Zulu/Xhosa short-text classification (F1 ~0.77-0.79) than every other language (≥0.98) — the two languages are closely related with substantial shared vocabulary. Treat predictions between these two with reduced confidence on short input.

---

## LIDLite608 / LIDNeural608 — 608 languages

African-first identification across **608 languages and 36 scripts**, plus a noise class for numbers, URLs and code. Both models share their languages, labels and two reading modes; pick by speed against short-text accuracy.

| | `LIDLite608` | `LIDNeural608` |
|---|---|---|
| Engine | fastText (CPU only) | ModernBERT / mmBERT, 140M parameters |
| Size | 37 MB | ~560 MB |
| Speed | ~6,800 texts/s on one CPU thread | ~1,100 texts/s on an A100 |
| Held-out test, accuracy | 0.858 | 0.867 |
| 1–5 word input, traffic mode (web-weighted) | 0.829 | 0.906 |
| Everyday phrases, traffic mode | 0.838 | 0.943 |
| Install | `olaverse[lid]` | `olaverse[deeplearning]` |

Figures are from the model cards: [olaverse/lid-lite-608](https://huggingface.co/olaverse/lid-lite-608) · [olaverse/lid-neural-608](https://huggingface.co/olaverse/lid-neural-608). Use both: run `LIDLite608` as a fast first pass and send only short or uncertain inputs to `LIDNeural608`.

```python
from olaverse import LIDLite608, LIDNeural608

lite = LIDLite608()
lite.predict("Ẹ kú àárọ̀, ṣé dáadáa ni?")     # → 'yor_Latn'
lite.predict_proba("Habari za asubuhi", top_k=3)
# → {'swh_Latn': 0.916, 'swc_Latn': 0.084, 'hau_Latn': 0.0}

neural = LIDNeural608()                         # CUDA if available (bf16), else CPU
neural.predict_batch(["Habari za asubuhi", "Mo fẹ́ lọ sí ọjà lónìí", "Bonjour mon ami"])
# → ['swh_Latn', 'yor_Latn', 'fra_Latn']
```

**Labels** are ISO 639-3 plus ISO 15924 script: `yor_Latn`, `srp_Cyrl`. `zxx_Zxxx` marks numbers, URLs, code and other non-language text. The 5- and 25-language classes return bare codes like `yor`; use `label.split("_")[0]` for the code alone.

**Two modes** (`mode=`):

| `mode=` | Use for |
|---|---|
| `"coverage"` *(default)* | Every language equally likely. Corpus building and mining low-resource text; best per-language accuracy. |
| `"traffic"` | Scores shifted by each language's real-world frequency, so short or ambiguous input leans towards common languages. User input, chat and routing. |

```python
LIDLite608(mode="traffic").predict("Bonjour mon ami")      # → 'fra_Latn'
LIDLite608().predict("Bonjour mon ami")                    # → 'dhv_Latn'  (coverage: two words are ambiguous)
LIDNeural608(mode="traffic").predict_batch(["Good morning", "Thank you very much"])
# → ['eng_Latn', 'eng_Latn']
```

Notes:

- **Short text.** Very short input is hard for every model here, the lite one most of all (`LIDLite608` labels `Good morning` as `ceb_Latn`, even in traffic mode). Use `LIDNeural608` for one-to-five-word input.
- **Long text.** `LIDNeural608` reads up to 512 tokens (about 200 words) and truncates the rest. Split long documents into ~200-word chunks and aggregate; this also catches pages that switch language partway through.
- **Input is collapsed to one line.** Newlines and runs of whitespace become single spaces (fastText reads one line, and both models were trained on it). Empty or whitespace-only text raises `ValueError`.
- **`device=`** (`LIDNeural608` only): `"cpu"`, `"cuda"`, `"mps"`, or `None`/`"auto"` to pick CUDA when available. The weights load in bfloat16 on CUDA and float32 elsewhere.
- **`transformers>=5.14`** is what the `LIDNeural608` model card requires; an older version can fail while loading, and the error says so.

---

## LIDNeural5_1 — Nigerian-only

A compact (~31M parameter) classifier built on [`mist-encoder-base-ng`](https://huggingface.co/olaverse/mist-encoder-base-ng), covering only Yoruba, Igbo, Hausa, and Nigerian Pidgin.

**Model Card**: [olaverse/lid-neural-5.1](https://huggingface.co/olaverse/lid-neural-5.1)

```python
from olaverse import LIDNeural5_1

detector = LIDNeural5_1()
detector.predict("Ina kwana?")   # → 'Hausa'
```

!!! danger "No English class"
    Out-of-set input (English or any other language) will be **confidently mislabelled**, most often as Nigerian Pidgin. If your input may include English, use `LIDNeural5` or the 25-language models instead.

---

## Benchmarks

| Model | Size | Speed | Macro F1 |
|---|---|---|---|
| `LIDLite5` | 1.1 MB | 0.014 ms | 98.12% |
| `LIDNeural5` | 484 MB | 13.3 ms | 98.96% |

Per-language precision/recall and the 25-language numbers: **[Benchmarks →](../benchmarks.md)**

---

## Applications

- ✅ **Chat & support routing** — send Hausa messages to Hausa-speaking agents
- ✅ **Corpus building** — label scraped text by language before training
- ✅ **Pipeline routing** — `Diacritizer(model="auto")` and other language-conditional steps
- ✅ **Content moderation** — apply the right language-specific rules
- ✅ **Search** — pick per-language analyzers and tokenizers at query time

---

## API Reference

Full class reference: [NLP & Tokenization → Language Detection](../nlp.md#language-detection)
