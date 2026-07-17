# DiacNet

**DiacNet is a multilingual diacritization model family that restores missing accents, tones, and language-specific characters — for NLP, OCR, TTS, and search systems.**

```text
Input:   se eranko naa si gbo o?
Output:  ṣé ẹranko náà sì gbọ́ ọ?
```

Most digital text in tonal and accented languages is typed without diacritics — keyboards, OCR, and legacy systems strip them. But diacritics carry meaning: Yoruba `ogun` can mean *war*, *twenty*, *medicine*, or the deity *Ògún* depending on tone marks. DiacNet puts them back.

---

## Supported Languages

| Model scope | Languages |
|---|---|
| Dedicated models | Yoruba, Igbo |
| `diacnet-1.0` (multilingual) | Yoruba, Igbo, Hausa, Vietnamese, Polish, Turkish, Portuguese, Spanish, French, Italian |

---

## Available Models

| Model ID | Language | Method | Speed | Accuracy | Size |
|---|---|---|---|---|---|
| `diacnet-yor-viterbi` | Yoruba | Viterbi n-gram | ⚡ Fast | Good | ~7 MB |
| `diacnet-yor-db` | Yoruba (dot-below only) | KNN backoff | ⚡ Fast | Dot-below focused | ~2 MB |
| `diacnet-yor` | Yoruba | BiLSTM | Medium | 93.35% char | 2.4 MB |
| `diacnet-yor-x` | Yoruba (full) | XLM-RoBERTa | Slow | 82.46% word | 503 MB |
| `diacnet-ig` | Igbo | KNN backoff | ⚡ Fast | Good | ~3 MB |
| `diacnet-1.0` | 10 languages | ByT5 seq2seq | Slow | ~0.02 median CER | ~300 MB |

### Which DiacNet should I use?

| Need | Model |
|---|---|
| Fast Yoruba, CPU-only | `diacnet-yor-viterbi` |
| Highest Yoruba accuracy | `diacnet-yor-x` |
| Igbo | `diacnet-ig` |
| 10 languages, one model | `diacnet-1.0` |
| Automatic routing (Yoruba/Igbo) | `auto` |

---

## Installation

```bash
pip install olaverse                 # Viterbi/KNN/BiLSTM models — CPU, no extras
pip install olaverse[deeplearning]   # adds diacnet-yor-x and diacnet-1.0
```

---

## Usage

### Quick functions

```python
from olaverse import diacritize_yoruba, diacritize_yoruba_dot_below, diacritize_igbo

diacritize_yoruba("Ojo lo si oja lana")
# → 'Òjó lọ sí ọjà lana'

diacritize_yoruba_dot_below("Ojo lo si oja")
# → 'Ọjọ lo si ọja'

diacritize_igbo("Kedu ka i mere")
# → 'Kedụ ka ị mere'
```

### The `Diacritizer` class

```python
from olaverse.nlp import Diacritizer

# Pick a specific model
d = Diacritizer(model="diacnet-yor-viterbi")
d.restore("Ojo lo si oja lana")
# → 'Òjó lọ sí ọjà lana'

# Automatic language routing — LIDLite5 detects, then routes
d_auto = Diacritizer(model="auto")
d_auto.restore("Kedu ka i mere")   # Igbo detected → diacnet-ig
# → 'Kedụ ka ị mere'
```

### Multilingual — `diacnet-1.0`

One joint ByT5 model, 10 languages, selected via `lang=`:

```python
d = Diacritizer(model="diacnet-1.0", lang="fr")
d.restore("cest fini")
# → "c'est fini"

d_yo = Diacritizer(model="diacnet-1.0", lang="yo")
d_yo.restore("se eranko naa si gbo o?")
# → 'ṣé ẹranko náà sì gbọ́ ọ?'
```

Supported `lang=` codes: `"yo", "vi", "ig", "ha", "pl", "tr", "pt", "es", "fr", "it"`.

**Model Card**: [olaverse/diacnet-1.0](https://huggingface.co/olaverse/diacnet-1.0)

---

## Performance

- `diacnet-1.0` reaches a **median CER of ~0.02** across its 10 languages on [DiacBench](../datasets.md).
- Yoruba is the hardest language for the multilingual model (median CER 0.110) — genuine tonal ambiguity, since the same base letters can carry multiple valid tone patterns. For peak Yoruba accuracy, prefer the dedicated `diacnet-yor-x`; for speed, `diacnet-yor-viterbi`.
- Benchmark it yourself — the [DiacBench dataset](../datasets.md) ships ~1,000 test pairs per language:

```python
from olaverse import load_dataset
from olaverse.nlp import Diacritizer

bench = load_dataset("diacbench", "yo", split="test")   # olaverse[data]
d = Diacritizer(model="diacnet-yor-viterbi")
restored = d.restore(bench[0]["input"])
```

Full published numbers: **[Benchmarks →](../benchmarks.md)**

---

## Applications

- ✅ **OCR correction** — restore diacritics that scanners and OCR engines drop
- ✅ **Text-to-speech preprocessing** — tone marks are the hardest front-end step of Yoruba TTS; DiacNet solves it
- ✅ **Language learning** — show learners correctly marked text
- ✅ **Search normalization** — index and match diacritized and plain text consistently
- ✅ **Digital archives** — repair legacy text corpora typed without diacritics
- ✅ **Translation pipelines** — give MT systems unambiguous, fully marked input

---

## Roadmap

- **DiacNet 1.1** — improved Yoruba restoration, better Igbo accuracy, lower CER, faster inference *(planned)*
- More African languages
- Streaming/batched inference API

See the full **[project roadmap →](../roadmap.md)**.

---

## API Reference

Full class/function reference: [NLP & Tokenization → Diacritization](../nlp.md#diacritization)
