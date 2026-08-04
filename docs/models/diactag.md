# DiacTag

**A diacritic model that cannot corrupt your text.** DiacTag restores accents and
tone marks in 10 languages by classifying each character rather than generating
new text — so the output is guaranteed to be the input with marks added, and
nothing else.

```text
Input:   se eranko naa si gbo o?
Output:  ṣé ẹranko náà sì gbọ́ ọ?
```

---

## Why a tagger and not seq2seq

[`diacnet`](diacnet.md) treats diacritization as translation: bare text in,
marked text out, generated token by token. It works, mostly — but "mostly" hides
something. On Hausa, only **94.7%** of `diacnet-1.1` outputs still stripped back
to their input. The other 5.3% weren't mis-accented, they were *different text*:
words dropped, clauses rewritten, punctuation invented.

That is not a tuning problem. A generative decoder can emit any token at any
position, so nothing in the architecture prevents it.

DiacTag reframes the task. The output has exactly as many characters as the
input, in the same order, with the same base letters — only the marks change. So
don't generate; **classify**. For each character, predict which diacritics it
carries and copy the base character through.

```
input      s    e    r    a    n    k    o
           │    │    │    │    │    │    │
      ┌─────────────────────────────────────┐
      │      character transformer          │
      └─────────────────────────────────────┘
  SHAPE   dot   ·    ·    ·    ·    ·    ·
  TONE     ·    ·    ·  acute  ·    ·    ·
           │    │    │    │    │    │    │
output     ṣ    e    r    á    n    k    o
```

The base character is never predicted, so `strip(output) == strip(input)` holds
**by construction** — for a trained model, an untrained one, or the int8 export.
The SDK asserts it on every call rather than assuming it.

| | `diacnet-1.1` (seq2seq) | `diactag-1.0` (tagger) |
|---|---|---|
| structural compliance | measured, ~94.7% on Hausa | **1.0000 by construction** |
| confidence | sequence-level | **per character, calibrated** |
| language detection | via a `<auto>` prefix token | **built-in LID head** |
| inference | autoregressive | one forward pass |
| parameters | 580M | **37.6M** |
| CPU serving | impractical | **the default** |
| typo correction | possible in principle | **impossible** — see below |

---

## Available model

| Model ID | Languages | Method | Backend | Size | DER |
|---|---|---|---|---|---|
| `diactag-1.0` | 10 | character transformer, 12 layers | PyTorch | ~150 MB | 0.0132 |
| `diactag-1.0` | 10 | same, int8 quantized | ONNX | **38 MB** | 0.0135 |

Yoruba, Igbo, Hausa, Vietnamese, Polish, Turkish, Portuguese, Spanish, French,
Italian.

**Model Card**: [olaverse/diactag-1.0](https://huggingface.co/olaverse/diactag-1.0)

---

## Installation

```bash
pip install olaverse[deeplearning]    # PyTorch checkpoint
pip install olaverse[onnx]            # adds the int8 ONNX backend
```

---

## Usage

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="diactag-1.0", lang="yo")
d.restore("se eranko naa si gbo o?")
# → 'ṣé ẹranko náà sì gbọ́ ọ?'
```

`lang=` accepts ISO-639-1 (`"yo"`) or ISO-639-3 (`"yor"`).

### Automatic language detection

Leave `lang` out and the model's own LID head decides. It costs almost nothing —
DER 0.0132 with the language supplied, 0.0133 with it detected — and most
integrations don't reliably know the input language.

```python
d = Diacritizer(model="diactag-1.0")

d.restore("Co ay rat dam dang")           # → 'Cô ấy rất đảm đang'
d.restore("El nino esta en la casa")      # → 'El niño está en la casa'

d.detect_language("Lodz jest piekna")     # → ('pol', 0.9999)
```

### Per-character confidence and abstention

The joint shape × tone distribution gives a calibrated probability per
character (temperature-fitted on validation, T = 1.14). Set a threshold and
characters below it are left exactly as the user typed them.

| threshold | coverage | DER on committed characters |
|---|---|---|
| 0.00 | 100.0% | 0.0132 |
| 0.90 | 97.1% | 0.0039 |
| 0.99 | 91.9% | 0.0008 |

At 0.90, **97% of characters are restored at 99.6% accuracy** and the rest are
flagged. A wrong tone mark changes meaning; a missing one is merely incomplete.

```python
d = Diacritizer(model="diactag-1.0", lang="yor", min_confidence=0.9)
d.restore("se eranko naa si gbo o?")

# The threshold is per-request too — one loaded model can serve a CMS pre-fill
# and a legal pipeline at different points on the same curve.
d.restore(text, min_confidence=0.99)
```

Route the uncertain characters to review:

```python
text, details = d.restore("se eranko naa", return_details=True)
review = [c for c in details if c.confidence < 0.9]

for c in details:
    c.char          # the emitted character
    c.confidence    # calibrated probability
    c.abstained     # left as the user typed it
    c.protected     # inside a URL/email/handle/identifier
```

### CPU serving with ONNX

Three times faster and four times smaller for +0.03pp DER. Compliance stays
`1.0000` under quantisation — the guarantee is architectural, not a property of
numeric precision.

```python
d = Diacritizer(model="diactag-1.0", lang="yor", onnx=True)
d.restore("se eranko naa si gbo o?")
```

Language auto-detection works on this backend too — it reads the same LID head
and agrees with PyTorch to four decimal places.

!!! note "Older exports have no LID head"
    Exports published before 2026-08-04 emit `shape_logits` and `tone_logits`
    only. The SDK reads the capability off the graph, so against one of those
    `lang=` becomes required and `restore()` raises without it, rather than
    silently treating your input as Yoruba.

### Lexicon reranking — opt-in, off by default

If the model emits a non-word whose stripped form has attested variants,
rescore the candidates under the model's own per-character distribution and
take the best. It never invents a word — it only chooses among attested forms,
and only when the model's own output wasn't one.

```python
d = Diacritizer(model="diactag-1.0", lang="yor", use_lexicon=True)
```

!!! warning "Measure it on your data before enabling it"
    On diacbench it cuts non-word outputs by **27%** and raises Yorùbá DER by
    **15%** (0.0836 → 0.0961). It trades a metric a reader notices against one
    they don't, and which way that trade falls depends on your copy.

    The cause is the density gating that makes the model good in the first
    place. Restricting the lexicon to well-marked text shrank the Yorùbá
    vocabulary from 86k forms to **18,436** — the harshest cut of any language,
    because 89% of the Yorùbá corpus was under-marked. So "not in the lexicon"
    frequently means "rare or inflected word we didn't keep" rather than "wrong
    spelling", and correct output gets overwritten.

    For comparison, the same gate left Igbo at 62,186 forms and Italian at
    44,977, so the effect is far weaker there.

### Documents, not just sentences

Unlike `diacnet`, no sentence splitting is needed. Overlapping windows are
planned across the input and only the centre of each is trusted, so every
character is predicted with context on both sides. The window plan is proved to
partition the input exactly.

```python
d.restore(open("article.txt").read())     # just works
```

### Protected spans

URLs, emails, `@handles`, inline code, `CONSTANT_NAMES` and bare domains pass
through untouched — and were excluded from the training loss. An accent inside a
URL is never correct, and it is the error a reader notices first.

```python
d.restore("Visit https://ile-ife.com or email ade@ola.ng for eniyan")
# → 'Visit https://ile-ife.com or email ade@ola.ng for ènìyàn'
```

### Marks you typed are never deleted

`respect_existing` is on by default: a mark already present in the input is
treated as your intent and preserved. 30% of training examples kept a random
subset of their marks, so half-corrected input is in-distribution rather than
an edge case.

---

## Performance

[diacbench](../datasets.md), 1000 sentences per language.

| lang | DER | shape | tone | exact | compliance |
|---|---|---|---|---|---|
| ita | 0.0002 | 0.0000 | 0.0002 | 0.991 | 1.0000 |
| fra | 0.0012 | 0.0003 | 0.0009 | 0.952 | 1.0000 |
| tur | 0.0016 | 0.0016 | 0.0000 | 0.961 | 1.0000 |
| por | 0.0019 | 0.0006 | 0.0014 | 0.925 | 1.0000 |
| pol | 0.0022 | 0.0018 | 0.0003 | 0.935 | 1.0000 |
| spa | 0.0022 | 0.0001 | 0.0021 | 0.917 | 1.0000 |
| hau | 0.0041 | 0.0040 | 0.0001 | 0.741 | 1.0000 |
| ibo | 0.0122 | 0.0110 | 0.0013 | 0.483 | 1.0000 |
| vie | 0.0164 | 0.0073 | 0.0124 | 0.650 | 1.0000 |
| yor | 0.0836 | 0.0203 | 0.0695 | 0.084 | 1.0000 |
| **all** | **0.0132** | 0.0055 | 0.0086 | 0.764 | **1.0000** |

Against the model it replaces: Yoruba **0.2006 → 0.0836** (58% lower), Hausa
**0.0593 → 0.0041** (93%) — though much of the Hausa gain is the compliance
guarantee rather than better modelling, since a large share of the old error was
text corruption rather than wrong accents.

`shape_DER` counts errors of letter identity; `tone_DER` counts errors of pitch
or stress. They are separate heads because which mark is which is
language-dependent — U+0323 dot-below is *shape* in Yoruba (`ẹ` and `e` are
different letters) and *tone* in Vietnamese (nặng). Reporting them apart is what
showed that four fifths of the remaining Yoruba error is a single failure mode.

### Versus frontier LLMs

300 sentences per language, matching how the baselines were run.

| lang | `diactag-1.0` | Claude Sonnet 4.5 | GPT-4o-mini |
|---|---|---|---|
| yor | **0.0933** | 0.1913 | 0.2811 |
| ibo | **0.0110** | 0.0427 | 0.1277 |
| hau | **0.0041** | 0.0178 | 0.1432 |
| vie | 0.0166 | **0.0107** | 0.0399 |
| fra | **0.0014** | 0.0052 | 0.0023 |

Best on 7 of 10; Vietnamese and Portuguese genuinely lose. Note that those LLM
numbers are the *charitable* ones — they come from a harness that discards any
output that no longer strips back to the input. Raw, against the floor of simply
copying the input unchanged, Claude scores 0.3509 on Hausa against a floor of
0.0236: on three languages a frontier model makes the text **worse than doing
nothing**. That fallback harness is this architecture, reimplemented externally.
Here there is nothing to discard.

Full published numbers: **[Benchmarks →](../benchmarks.md)**

---

## Deployment

| backend | chars/s | p50 latency | size | DER |
|---|---|---|---|---|
| PyTorch CPU | 105 | 591 ms | 150 MB | 0.0105 |
| ONNX fp32 | 202 | 286 ms | 150.9 MB | 0.0105 |
| **ONNX int8** | **244** | **200 ms** | **38.3 MB** | 0.0108 |

A 38MB artifact at 244 characters per second on one CPU core takes the GPU off
the serving bill entirely.

---

## Limitations

- **No typo correction.** The architecture cannot insert or delete characters,
  so it cannot fix `Ile → Ilé` and `teh → the` in one pass. That is the price of
  the guarantee. The per-character confidence is a natural trigger for a
  separate corrector: characters where the tagger is unsure are exactly where a
  spelling issue is likely.
- **Yoruba is still hard.** DER 0.0836, 83% of it tone *direction* — the model
  knows a mark belongs there and picks the wrong one. Sentence-level exact match
  is 0.084, so 92 of every 100 Yoruba sentences contain at least one wrong mark.
- **Igbo and Hausa tone numbers are not achievements.** `tone_DER` of 0.0013 and
  0.0001 looks superb and means little: those orthographies barely write tone.
- **Dense input degrades.** The Polish pangram *Zażółć gęślą jaźń* has nine
  times the density of real Polish and the model misses six characters, despite
  a Polish DER of 0.0022.
- **Some errors are irreducible.** `Viaggio` and `Viaggiò` are both valid
  Italian and the stripped form contains no information distinguishing them.
- **Fixed label space.** Adding a language with new marks invalidates existing
  checkpoints. The label space carries a `SPEC_VERSION` that is checked on load,
  so a mismatch fails loudly rather than silently.

---

## Which diacritization model should I use?

| Need | Model |
|---|---|
| Output must never differ from input except in marks | **`diactag-1.0`** |
| Yoruba, Igbo or Hausa accuracy | **`diactag-1.0`** |
| CPU-only serving at scale | **`diactag-1.0`** (`onnx=True`) |
| Confidence scores / human review routing | **`diactag-1.0`** |
| Vietnamese or Portuguese peak accuracy | [`diacnet-1.1`](diacnet.md) |
| Fast Yoruba with no deep-learning extra | [`diacnet-yor-viterbi`](diacnet.md) |

---

## API Reference

Full class/function reference: [NLP & Tokenization → Diacritization](../nlp.md#diacritization)
