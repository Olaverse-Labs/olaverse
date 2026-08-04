# DiacNet

**DiacNet is a multilingual diacritization model family that restores missing accents, tones, and language-specific characters — for NLP, OCR, TTS, and search systems.**

```text
Input:   se eranko naa si gbo o?
Output:  ṣé ẹranko náà sì gbọ́ ọ?
```

Most digital text in tonal and accented languages is typed without diacritics — keyboards, OCR, and legacy systems strip them. But diacritics carry meaning: Yoruba `ogun` can mean *war*, *twenty*, *medicine*, or the deity *Ògún* depending on tone marks. DiacNet puts them back.

!!! tip "For most new work, start with DiacTag"
    [`diactag-1.0`](diactag.md) is a per-character tagger rather than a seq2seq
    model. It cannot change, insert or delete a base character, it scores every
    character, it runs on CPU at 38MB, and it beats DiacNet on 9 of 10
    languages. DiacNet remains the right choice for Vietnamese and Portuguese
    peak accuracy, and the small Viterbi/KNN models remain the fastest way to
    do Yoruba or Igbo without a deep-learning install.

---

## Supported Languages

| Model scope | Languages |
|---|---|
| Dedicated models | Yoruba, Igbo |
| `diacnet-1.0` / `diacnet-1.1` (multilingual) | Yoruba, Igbo, Hausa, Vietnamese, Polish, Turkish, Portuguese, Spanish, French, Italian |

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
| `diacnet-1.1` | 10 languages | ByT5 seq2seq | Slow | 0.0002–0.2006 DER | ~1.1 GB |

### Which DiacNet should I use?

| Need | Model |
|---|---|
| Fast Yoruba, CPU-only | `diacnet-yor-viterbi` |
| Highest Yoruba accuracy | [`diactag-1.0`](diactag.md), then `diacnet-yor-x` |
| Igbo | [`diactag-1.0`](diactag.md), or `diacnet-ig` with no extras |
| 10 languages, one model | [`diactag-1.0`](diactag.md) |
| Vietnamese, Turkish, Polish, Italian via seq2seq | `diacnet-1.1` |
| Yoruba, Igbo or Hausa via seq2seq | `diacnet-1.0` |
| Automatic routing (Yoruba/Igbo) | `auto` |

---

## Installation

```bash
pip install olaverse                 # Viterbi/KNN/BiLSTM models — CPU, no extras
pip install olaverse[deeplearning]   # adds diacnet-yor-x, diacnet-1.0, diacnet-1.1
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
d.restore("Le cafe est tres chaud, mais il prefere le the.")
# → 'Le café est très chaud, mais il préfère le thé.'

d_yo = Diacritizer(model="diacnet-1.0", lang="yo")
d_yo.restore("se eranko naa si gbo o?")
# → 'ṣé ẹranko náà sì gbọ́ ọ?'
```

Supported `lang=` codes: `"yo", "vi", "ig", "ha", "pl", "tr", "pt", "es", "fr", "it"`.

### `diacnet-1.1` — same architecture, larger corpus

v1.1 is the same ByT5 model retrained on a much larger web-sourced corpus. It is
a **large improvement on 5 languages and a regression on 3**, so it does not
simply supersede v1.0 — pick per language:

| lang | DER 1.0 | DER 1.1 | verdict |
|---|---|---|---|
| vie | 0.1264 | **0.0460** | 1.1 — 2.7× better |
| tur | 0.0447 | **0.0068** | 1.1 — 6.6× better |
| pol | 0.0357 | **0.0058** | 1.1 — 6.2× better |
| ita | 0.0015 | **0.0002** | 1.1 — 7.5× better |
| por | 0.0072 | **0.0031** | 1.1 better |
| spa | 0.0084 | **0.0081** | ~equal |
| fra | **0.0038** | 0.0053 | mixed (1.1 has lower WER) |
| ibo | **0.0359** | 0.0508 | mixed (1.1 has lower WER) |
| hau | **0.0383** | 0.0593 | mixed (1.1 has lower WER) |
| yor | **0.1554** | 0.2006 | 1.0 better |

The cause is measured, not speculative: v1.0 trained on ~2,000 well-tone-marked
Yoruba passages (diacritic density 0.565), while v1.1's larger corpus averages
0.223 — it contains far more Yoruba text, but most of it omits tone marks, so
the model learned to omit them too. More data at lower annotation quality lost
to less data at higher quality.

[`diactag-1.0`](diactag.md) is the fix for that regression: it scores **0.0836**
on Yoruba and **0.0041** on Hausa by gating which sentences are allowed to
supply diacritic supervision.

```python
d = Diacritizer(model="diacnet-1.1", lang="vi")
d.restore("Toi khong biet tieng Viet")
# → 'Tôi không biết tiếng Việt'
```

**Model Card**: [olaverse/diacnet-1.1](https://huggingface.co/olaverse/diacnet-1.1)

### Long text is segmented automatically

`diacnet-1.0` was trained on sentence-length input (median 58 bytes). The SDK
handles that for you: multi-sentence text is split on sentence boundaries,
restored a sentence at a time, and rejoined.

This matters. On a 358-character French paragraph, restoring it in one pass
returns **235 characters** — it truncates mid-sentence and drops the tail
entirely. Segmented, all 358 characters come back correct.

```python
# Default: segments automatically
d = Diacritizer(model="diacnet-1.0", lang="fr")
d.restore(long_paragraph)

# One pass, no segmentation
Diacritizer(model="diacnet-1.0", lang="fr", split_sentences=False)

# Your own segmentation
Diacritizer(model="diacnet-1.0", lang="fr", splitter=my_splitter)
```

!!! warning "Very short fragments still degenerate"
    Single words fall *below* the trained input length and misbehave:
    repetition loops (`"el nino"` → `'el niño\nel niño\nel niño…'`), changed
    inflection (`"nino"` → `'niños'`), invented punctuation (`"citta"` →
    `'città?'`), or another language's diacritics (`"cafe"` with `lang="fr"` →
    `'cafẹ́'`, a Yoruba dot-below). Pass whole sentences.

    It restores **diacritics only** — it does not insert apostrophes, so
    `"cest fini"` will not become `"c'est fini"`.

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

- ✅ **DiacNet 1.1** — shipped; large gains on Vietnamese, Turkish, Polish and Italian, a regression on Yoruba
- ✅ **DiacTag 1.0** — shipped; the tone-marking regression fixed, plus a structural compliance guarantee. [DiacTag →](diactag.md)
- More African languages
- Streaming/batched inference API

See the full **[project roadmap →](../roadmap.md)**.

---

## API Reference

Full class/function reference: [NLP & Tokenization → Diacritization](../nlp.md#diacritization)
