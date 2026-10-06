# DiacNet

**DiacNet is a multilingual diacritization model family that restores missing accents, tones, and language-specific characters — for NLP, OCR, TTS, and search systems.**

```text
Input:   se eranko naa si gbo o?
Output:  ṣé ẹranko náà sì gbọ́ ọ?
```

Most digital text in tonal and accented languages is typed without diacritics — keyboards, OCR, and legacy systems strip them. But diacritics carry meaning: Yoruba `ogun` can mean *war*, *twenty*, *medicine*, or the deity *Ògún* depending on tone marks. DiacNet puts them back.

!!! tip "Which generation?"
    **`diacnet-2.0` / `diacnet-mini-2.0`** (new in v0.4.0) add Arabic, `<auto>`
    language detection and meaning hints, and align their output onto your text by
    default so they no longer change letters — see
    [diacnet-2.0](#diacnet-20-and-diacnet-mini-20). Per the model cards,
    `diacnet-2.0` has a lower error rate than [`diactag-2.0`](diactag.md) on 8 of
    the 10 Latin-script languages.

    [`diactag`](diactag.md) is a per-character tagger rather than a seq2seq model.
    It cannot change, insert or delete a base character, it scores every
    character, and it runs on CPU at 38MB. It beat `diacnet-1.x` on 9 of 10
    languages, and stays the CPU-native choice. The small Viterbi/KNN models remain
    the fastest way to do Yoruba or Igbo without a deep-learning install.

---

## Supported Languages

| Model scope | Languages |
|---|---|
| Dedicated models | Yoruba, Igbo |
| `diacnet-1.0` / `diacnet-1.1` (multilingual) | Yoruba, Igbo, Hausa, Vietnamese, Polish, Turkish, Portuguese, Spanish, French, Italian |
| `diacnet-2.0` / `diacnet-mini-2.0` (multilingual) | the ten above, plus **Arabic** |

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
| `diacnet-2.0` | 11 languages | ByT5-base text-to-text, 582M | Slow (GPU recommended) | mean DER 0.0117 aligned | 2.3 GB |
| `diacnet-mini-2.0` | 11 languages | ByT5, 300M | Slow (GPU recommended) | mean DER 0.0141 aligned | 1.2 GB |

### Which DiacNet should I use?

| Need | Model |
|---|---|
| Fast Yoruba, CPU-only | `diacnet-yor-viterbi` |
| Highest Yoruba accuracy | [`diactag-1.0`](diactag.md), then `diacnet-yor-x` |
| Igbo | [`diactag-1.0`](diactag.md), or `diacnet-ig` with no extras |
| 10 languages, one model | [`diactag-1.0`](diactag.md) |
| Vietnamese, Turkish, Polish, Italian via seq2seq | `diacnet-1.1` |
| Yoruba, Igbo or Hausa via seq2seq | `diacnet-1.0` |
| Arabic, 11 languages, meaning hints, `<auto>` | `diacnet-2.0` (`diacnet-mini-2.0` at half the size) |
| Automatic routing (Yoruba/Igbo) | `auto` |

---

## Installation

```bash
pip install olaverse                 # Viterbi/KNN/BiLSTM models — CPU, no extras
pip install olaverse[deeplearning]   # adds diacnet-yor-x, diacnet-1.0/1.1, diacnet-2.0/mini-2.0
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

### diacnet-2.0 and diacnet-mini-2.0

ByT5 text-to-text models for 11 languages including Arabic (**New in v0.4.0**).
`diacnet-2.0` is 582M parameters, `diacnet-mini-2.0` 300M with the same interface.
Weights download from Hugging Face on first use (2.3 GB and 1.2 GB).

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="diacnet-2.0", lang="yor", device="cuda")   # bf16 on GPU
d.restore("O so fun ara re pe oun ko ni isoro kankan.")
# → 'Ó sọ fún ara rẹ̀ pé òun kò ní ìṣòro kankan.'

d = Diacritizer(model="diacnet-2.0")                 # no lang= → <auto>
d.restore("No pude sujetarme a la cuerda mas tiempo.")
# → 'No pude sujetarme a la cuerda más tiempo.'
```

`lang=` takes ISO-639-3 or ISO-639-1 (`yor`/`yo`, `ara`/`ar`, …), `"auto"`, or
`"ara-nocase"`; leave it out for `<auto>` (the 1.x models fall back to Yoruba
instead). On `diacnet-2.0`, `<auto>` costs at most 0.0005 DER against an explicit
tag (model card).

**Arabic.** `lang="ara"` restores everything including the case ending on each
word's last letter; `case_endings=False` (or `lang="ara-nocase"`) leaves word-final
vowels off, shadda kept. Hamza letters are kept as typed.

```python
d.restore("وهذا قول مرغوب عنه .", lang="ara")
# → 'وَهَذَا قَوْلٌ مَرْغُوبٌ عَنْهُ .'
d.restore("وهذا قول مرغوب عنه .", lang="ara", case_endings=False)
# → 'وَهَذَا قَوْل مَرْغُوب عَنْه .'
```

**Meaning hints.** Some words take different marks for different meanings
(Vietnamese *ranh*: rảnh "free", rành "skilled", ranh "border"; Yorùbá *ogun*: war
or twenty). A hint, written in English, steers the choice. It is a
`{word: meaning}` dict, a list of `"word=meaning"` strings, or a ready-made
`"[g: ...]"` block, and goes into the prompt as `<tag> [g: word=meaning] text`
(several hints are joined with ` | `):

```python
s = "Chi ay chi that su ranh vao nhung buoi toi sau khi da cho con ngu say."
d.restore(s, lang="vie", hints={"ranh": "free (time)"})
# → 'Chị ấy chỉ thật sự rảnh vào những buổi tối sau khi đã cho con ngủ say.'
```

Hints help most where the sentence alone is not enough (Vietnamese, Turkish,
Polish); when a hint contradicts a clear context the model usually follows the
context. The characters `|`, `]` and a newline are not allowed inside a hint.

**Output alignment (`aligned=True`, the default).** A generative model can change
a letter, not just add marks. With alignment on, your own letters are kept and only
the model's marks are taken, so the output always strips back to your input. Set
`aligned=False` for the raw generation, which also repairs typos (swapped, missing
or doubled letters) but may change a letter in a few percent of sentences. The
model card reports Igbo raw DER 0.0926 against 0.0185 aligned, and Hausa 0.0485
against 0.0041, almost all of it from changed letters. Only *letters* are
protected: marks come from the model, so a mark you typed can still be replaced by
the model's choice (the model card reports 99.9–100% of given marks kept).

```python
from olaverse.nlp import align_marks        # the same function, standalone
align_marks("se eranko naa si gbo o", "ṣe ẹranko náà sì gbọ́ ọ?")
# → 'ṣe ẹranko náà sì gbọ́ ọ'      (the added "?" is dropped)
```

**Long text and batching.** Text is split into chunks of at most ~300 characters on
spaces (not sentence ends), decoded greedily, and rejoined with single spaces.
`restore_batch` pools the chunks of many texts, sorts them by length and decodes
them in batches padded to the longest, which is far faster on a GPU than one call
per text:

```python
d = Diacritizer(model="diacnet-2.0", device="cuda", batch_size=16)
d.restore_batch(texts, lang="yor")                          # one language for all
d.restore_batch(texts, lang=["yor", "ara", "auto"], hints=[None, None, None])
```

`device=` takes `"cpu"` (default for `Diacritizer`), `"cuda"`, `"mps"` or `"auto"`;
the weights load in bfloat16 on CUDA and float32 elsewhere. `hints=` in
`restore_batch` is one entry per text (or `None`) — it is not broadcast.

!!! note "`diacnet-1.0` / `1.1` are unchanged"
    The 1.x models keep their own prompt format, sentence splitting and raw output.
    They accept `aligned=True` if you want alignment, but it is off by default so
    existing output does not change. They have no Arabic and no hints.

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
- ✅ **DiacNet 2.0 / DiacNet-mini 2.0** — shipped; Arabic, meaning hints, `<auto>`, output alignment
- ✅ **DiacTag 1.0** — shipped; the tone-marking regression fixed, plus a structural compliance guarantee. [DiacTag →](diactag.md)
- More African languages
- Streaming/batched inference API

See the full **[project roadmap →](../roadmap.md)**.

---

## API Reference

Full class/function reference: [NLP & Tokenization → Diacritization](../nlp.md#diacritization)
