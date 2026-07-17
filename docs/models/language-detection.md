# Language Detection (LID)

**Five language-identification models, from a 1.1 MB pure-Python classifier to 25-language neural models — pick by coverage, latency, and accuracy.**

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
# → {'ibo': 0.993, 'pcm': 0.003, 'eng': 0.002, ...}

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
