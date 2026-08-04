# Benchmarks

All published Olaverse model numbers in one place. Where a figure isn't listed here, it hasn't been formally measured — we don't publish estimates.

---

## Language Detection

### 5-language models (Yoruba, Igbo, Hausa, Pidgin, English)

| Model | Size | Speed / sentence | Macro F1 |
|---|---|---|---|
| `LIDLite5` | 1.1 MB | **0.014 ms** | 98.12% |
| `LIDNeural5` | 484 MB | 13.3 ms | **98.96%** |

### LIDNeural5 — per-language breakdown

| Language | Precision | Recall | F1-Score |
|---|---|---|---|
| Yoruba (`yor`) | 99.60% | 99.60% | 99.60% |
| Hausa (`hau`) | 99.60% | 99.20% | 99.40% |
| Igbo (`ibo`) | 98.79% | 98.20% | 98.50% |
| Nigerian Pidgin (`pcm`) | 99.20% | 98.80% | 99.00% |
| English (`eng`) | 97.63% | 99.00% | 98.31% |
| **Overall (Macro)** | | | **98.96%** |

### 25-language models

| Model | Type | Short-text accuracy | Notes |
|---|---|---|---|
| `LIDLite25` | fastText, CPU | 97.3% | Sub-millisecond, ~5-10 MB per checkpoint |
| `LIDNeural25` | XLM-RoBERTa-base | **98.2%** | Requires `transformers`/`torch` |

Known weak spot: Zulu/Xhosa confusion on short text (F1 ~0.77-0.79 for that pair vs ≥0.98 everywhere else) — the languages share substantial vocabulary. Full per-language tables are on the model cards: [lid-lite-25](https://huggingface.co/olaverse/lid-lite-25) · [lid-neural-25.1](https://huggingface.co/olaverse/lid-neural-25.1) · [lid-neural-25.2](https://huggingface.co/olaverse/lid-neural-25.2).

---

## Diacritization

### Dedicated Yoruba/Igbo models

| Model | Method | Reported accuracy | Size |
|---|---|---|---|
| `diacnet-yor-viterbi` | Viterbi n-gram | Good (fast baseline) | ~7 MB |
| `diacnet-yor` | BiLSTM | **93.35% character accuracy** | 2.4 MB |
| `diacnet-yor-x` | XLM-RoBERTa | **82.46% word accuracy** | 503 MB |
| `diacnet-ig` | KNN backoff | Good (fast baseline) | ~3 MB |

### Multilingual models (10 languages)

DER = diacritic error rate over diacritic-eligible characters, on
[DiacBench](datasets.md). **Compliance** is the fraction of outputs that still
strip back to the input — anything below 1.0 means the model edited text it was
only asked to accent.

| lang | `diactag-1.0` | `diacnet-1.1` | `diacnet-1.0` |
|---|---|---|---|
| ita | 0.0002 | **0.0002** | 0.0015 |
| fra | **0.0012** | 0.0053 | 0.0038 |
| tur | **0.0016** | 0.0068 | 0.0447 |
| por | **0.0019** | 0.0031 | 0.0072 |
| pol | **0.0022** | 0.0058 | 0.0357 |
| spa | **0.0022** | 0.0081 | 0.0084 |
| hau | **0.0041** | 0.0593 | 0.0383 |
| ibo | **0.0122** | 0.0508 | 0.0359 |
| vie | **0.0164** | 0.0460 | 0.1264 |
| yor | **0.0836** | 0.2006 | 0.1554 |
| **all** | **0.0132** | — | — |
| **compliance** | **1.0000** | ~0.947 on hau | — |

`diactag-1.0` wins on 9 of 10 and ties on Italian. It also splits its error into
`shape_DER` (letter identity, 0.0055 overall) and `tone_DER` (pitch/stress,
0.0086) — separate heads, separate metrics, so a flattering aggregate cannot
hide a single dominant failure mode.

**Versus frontier LLMs**, 300 sentences per language: `diactag-1.0` is best on 7
of 10, losing Vietnamese and Portuguese to Claude Sonnet 4.5. Those LLM numbers
come from a harness that discards any output that no longer strips to the input;
scored raw, Claude reaches 0.3509 DER on Hausa against a copy-the-input floor of
0.0236 — on three languages a frontier model makes the text worse than doing
nothing.

**Abstention curve** (`diactag-1.0`, `min_confidence=`):

| threshold | coverage | DER on committed characters |
|---|---|---|
| 0.00 | 100.0% | 0.0132 |
| 0.90 | 97.1% | 0.0039 |
| 0.99 | 91.9% | 0.0008 |

**Throughput** on one CPU core: 105 chars/s PyTorch (150 MB), 244 chars/s int8
ONNX (38.3 MB) for +0.03pp DER. Compliance stays 1.0000 under quantisation.

Full per-language tables: [olaverse/diactag-1.0](https://huggingface.co/olaverse/diactag-1.0) ·
[olaverse/diacnet-1.1](https://huggingface.co/olaverse/diacnet-1.1) ·
[olaverse/diacnet-1.0](https://huggingface.co/olaverse/diacnet-1.0)

### Reproduce it yourself

DiacBench ships ~1,000 test pairs per language, one config per language (`es fr ha ig it pl pt tr vi yo`):

```python
from olaverse import load_dataset
from olaverse.nlp import Diacritizer

bench = load_dataset("diacbench", "yo", split="test")   # olaverse[data]
d = Diacritizer(model="diacnet-yor-viterbi")

restored = d.restore(bench[0]["input"])
reference = bench[0]["reference"]
```

---

## Tokenization Efficiency

Token count reduction vs GPT-4's cl100k tokenizer on native text:

| Tokenizer | Language | Efficiency |
|---|---|---|
| `otk-bpe-50k-yo` | Yoruba | **63% fewer tokens** |
| `otk-bpe-50k-ig` | Igbo | ~60% fewer tokens |
| `otk-bpe-50k-ha` | Hausa | ~58% fewer tokens |
| `otk-bpe-50k-pcm` | Nigerian Pidgin | ~55% fewer tokens |

All OTK-BPE tokenizers guarantee **0% out-of-vocabulary** via raw UTF-8 byte fallback. For the multilingual family (Swahili/Kinyarwanda/merged), fertility and entity-handling benchmarks are on the [otk-bpe model card](https://huggingface.co/olaverse/otk-bpe) — both improve monotonically with vocab size, making 150k the recommended default.

---

## MIST Inference Speed

| Variant | Params | Throughput |
|---|---|---|
| MIST-Mini-8B | 8B | ~63 tok/s |
| MIST-Mini-8B-Thinking | 8B | ~55 tok/s |
| MIST-1-70B | 70B | ~23 tok/s |
| MIST-1-140B | 140B | ~8 tok/s |

---

## LegalPeace vs Base Mistral-7B

| Benchmark | Improvement |
|---|---|
| Inference Speed | 10.3% faster |
| Contract Analysis | 32.6% faster |
| Case Predictions | 14.0% faster |

---

## Vision (Prism)

- `PrismDenoiser`: **+3-4 dB PSNR** on complex scenes (model-card benchmarks)
- `PrismSteganography`: **99.9%** clean bit-accuracy; **93.7%** average under distortion (worst case 62.5%)
- `PrismUpscaler`: not yet evaluated against standard academic benchmarks (Set5/Set14/BSD100/Urban100) — model-card comparisons are informal checks against a bicubic baseline
