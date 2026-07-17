# Tokenizers — OTK-BPE

**Byte-Level BPE tokenizers trained on native corpora for African languages — 0% out-of-vocabulary, up to 63% fewer tokens than GPT-4's tokenizer.**

```python
from olaverse import Tokenizer

tok = Tokenizer("yo")
ids = tok.encode("Ẹ kú àbọ̀")
tok.decode(ids)   # → 'Ẹ kú àbọ̀'
```

---

## Why not just use a general-purpose tokenizer?

General-purpose tokenizers like GPT-4's cl100k were trained mostly on English. On African languages they fall back to character-by-character (or byte-by-byte) splitting, which:

- **blows up sequence lengths** — more tokens per sentence means less effective context and higher API costs
- **fragments meaningful units** — subwords never align with morphemes
- **degrades model quality** — models learn from worse-shaped input

OTK-BPE learns proper subwords from native text, with raw UTF-8 byte fallback guaranteeing **0% OOV** on any input.

---

## Nigerian family — OTK-BPE-50k

**Model Card**: [olaverse/otk-bpe-50k](https://huggingface.co/olaverse/otk-bpe-50k)

| `lang=` | Language | Vocab | Efficiency vs GPT-4 |
|---|---|---|---|
| `"yo"` | Yoruba | 50,000 | **63% fewer tokens** |
| `"ig"` | Igbo | 50,000 | ~60% fewer tokens |
| `"ha"` | Hausa | 50,000 | ~58% fewer tokens |
| `"pcm"` | Nigerian Pidgin | 50,000 | ~55% fewer tokens |
| `"naija"` | Unified (all 4) | 50,000 | Balanced |

```python
from olaverse import Tokenizer

tok_yo = Tokenizer("yo")
tok_all = Tokenizer("naija")   # one vocab across all 4 languages

# Dataset preparation for fine-tuning
sentences = ["Bawo ni?", "Se daadaa ni?", "Mo dupe."]
all_ids = [tok_yo.encode(s) for s in sentences]
```

---

## Multilingual family — OTK-BPE

**Model Card**: [olaverse/otk-bpe](https://huggingface.co/olaverse/otk-bpe)

Swahili, Kinyarwanda, and a merged French + Kinyarwanda + English + Swahili vocabulary — each at three vocab sizes, through the same `Tokenizer` class:

| `lang=` | Languages | Vocab sizes |
|---|---|---|
| `sw-50k` / `sw-100k` / `sw-150k` | Swahili | 50k / 100k / 150k |
| `kin-50k` / `kin-100k` / `kin-150k` | Kinyarwanda | 50k / 100k / 150k |
| `merged-50k` / `merged-100k` / `merged-150k` | French + Kinyarwanda + English + Swahili | 50k / 100k / 150k |

**150k is the recommended default** — fertility and entity handling both improve monotonically from 50k → 100k → 150k in every benchmark on the model card. Step down only if embedding-table size is a hard constraint.

```python
tok = Tokenizer("sw-150k")
ids = tok.encode("Habari yako? Leo ni siku nzuri sana 😊")
tok.decode(ids)   # → exact round-trip, emoji included

tok_merged = Tokenizer("merged-150k")
```

---

## Installation

```bash
pip install olaverse   # all tokenizers included — no GPU, no extras
```

---

## Applications

- ✅ **LLM fine-tuning** — prepare datasets with language-appropriate subwords
- ✅ **Cost reduction** — fewer tokens per request against token-priced APIs
- ✅ **Training from scratch** — pair with `marco-style-pairs-multi` and other [olaverse datasets](../datasets.md)
- ✅ **Search indexing** — consistent subword units for lexical matching

---

## API Reference

Full class reference: [NLP & Tokenization → Tokenization](../nlp.md#tokenization-otk-bpe-50k)
