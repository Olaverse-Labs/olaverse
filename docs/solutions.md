# Solutions

How Olaverse components combine into production pipelines. Every snippet below runs against the current SDK.

---

## Speech AI

Diacritics are the hardest front-end step of tonal-language TTS — without tone marks, a synthesizer cannot pick the right pitch contour.

```text
Audio → ASR → Olaverse normalization → Diacritization → TTS
```

```python
from olaverse.nlp import TTSNormalizer, Diacritizer

normalizer = TTSNormalizer(lang="yo")
diacritizer = Diacritizer(model="diacnet-yor-viterbi")

text = "Dr. Ade lo si oja lana"                 # raw ASR output / user text
normalized = normalizer.normalize(text)          # "Dọ́kítà Ade lo si oja lana"
diacritized = diacritizer.restore(normalized)    # "Dọ́kítà Adé ló sí ọjà lànà"
```

For Nigerian Pidgin input, `NaijaNormalizer` first expands informal spellings (`2moro` → `tomorrow`, `tnx` → `thanks`) before standard normalization.

---

## OCR Correction

Scanners and OCR engines routinely drop diacritics. Restore them as a post-processing pass:

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="diacnet-1.0", lang="vi")   # 10 languages available
d.restore("Toi yeu tieng Viet")
```

Pair with `PrismDenoiser` to clean scans *before* OCR, and DiacNet to repair the text *after*.

---

## Search

Language-aware retrieval, end to end:

```python
from olaverse import LIDLite5, Embedder, Reranker

# 1. Detect the query language (0.014 ms — negligible overhead)
lang = LIDLite5().predict(query)

# 2. First-stage retrieval — cross-lingual embeddings for ha/yo/ig
embedder = Embedder()
query_vec = embedder.encode([query])[0]

# 3. Rerank the top-k candidates (English)
reranker = Reranker(size="22.7m")
ranked = reranker.rank(query, candidate_passages)
```

Diacritization also helps **search normalization** — index both plain and restored forms so `ogun` matches `Ògún`.

---

## Education

Language-learning products need correctly marked text; learners can't infer tones from bare letters.

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="auto")   # student types without diacritics
d.restore("Ojo lo si oja lana")
# → 'Òjó lọ sí ọjà lana'  — show the corrected form
```

---

## Customer Support

Route incoming messages by language before they hit an agent or a bot:

```python
from olaverse import LIDLite5

detector = LIDLite5()

lang = detector.predict("How far, wetin dey happen?")   # → 'pcm'
# route 'pcm'/'yor'/'ibo'/'hau' to the right queue or model
```

`predict_proba` gives you a confidence score for a human-fallback threshold.

---

## Translation Pipelines

MT systems produce better output from unambiguous input. Detect, normalize, and diacritize before translating:

```python
from olaverse import LIDLite5, NaijaNormalizer
from olaverse.nlp import Diacritizer

text = "abeg, 2moro na Sunday"

lang = LIDLite5().predict(text)                  # 'pcm'
clean = NaijaNormalizer().normalize(text)        # 'abeg, tomorrow na Sunday'
# → feed to your MT system; for yor/ibo input, add Diacritizer(model="auto")
```

---

## LLM Fine-tuning & Data Prep

Efficient tokenization plus public training data:

```python
from olaverse import Tokenizer, load_dataset

tok = Tokenizer("yo")                            # 63% fewer tokens than GPT-4
pairs = load_dataset("marco-style-pairs-multi", split="train")   # 25 languages
```

---

## Need something custom?

Custom language support, domain adaptation, and deployment help are available — see **[Commercial Support →](enterprise.md)**.
