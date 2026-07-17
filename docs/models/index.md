# Models

Every model Olaverse ships, in one place — grouped by task, with guidance on which variant to pick.

| Family | Task | Page |
|---|---|---|
| **DiacNet** | Diacritization (restore accents, tones, special characters) | [DiacNet →](diacnet.md) |
| **LID** | Language detection (5–25 languages) | [Language Detection →](language-detection.md) |
| **OTK-BPE** | Tokenization (Nigerian languages, Swahili, Kinyarwanda) | [Tokenizers →](tokenizers.md) |
| **Reranker / Embedder** | Retrieval, RAG, cross-lingual search | [Retrieval →](retrieval.md) |
| **MIST** | General-purpose LLMs (8B–140B) | [MIST →](mist.md) |
| **LegalPeace** | Legal contract reasoning | [LegalPeace →](legalpeace.md) |
| **Prism** | Image upscaling, denoising, steganography | [Prism →](prism.md) |

---

## Full Model Index

<div class="ov-model-table" markdown>

| Model | Task | Size | Speed | Install |
|---|---|---|---|---|
| `LIDLite5` | Language ID (5 langs) | 1.1 MB JSON | 0.014 ms | `olaverse` |
| `LIDNeural5` | Language ID (5 langs) | 484 MB | 13 ms | `olaverse[deeplearning]` |
| `LIDLite25` | Language ID (25 langs) | ~5-10 MB | <1 ms | `olaverse[lid]` |
| `LIDNeural25` | Language ID (25 langs) | ~500 MB | — | `olaverse[deeplearning]` |
| `LIDNeural5_1` | Language ID (4 Nigerian langs, no English) | ~120 MB | — | `olaverse[deeplearning]` |
| `MIST-Mini-8B` | General LLM | 15 GB | ~63 tok/s | `olaverse[deeplearning]` |
| `MIST-1-70B` | General LLM | 132 GB | ~23 tok/s | hosted or multi-GPU |
| `MIST-1-140B` | General LLM | 256 GB | ~8 tok/s | hosted or 2× H200 |
| `MIST-Mini-8B-Thinking` | Reasoning LLM | 15 GB | ~55 tok/s | `olaverse[deeplearning]` |
| `LegalPeace` | Legal reasoning | 7B (4-bit) | — | `olaverse[legal]` |
| `DiacNet` (5 Yoruba/Igbo variants) | Diacritization | 1 MB – 503 MB | — | `olaverse` / `[deeplearning]` |
| `diacnet-1.0` | Diacritization (10 langs) | ~300 MB | Slow | `olaverse[deeplearning]` |
| `OTK-BPE-50k` (5 Nigerian variants) | Tokenization | ~3 MB each | — | `olaverse` |
| `OTK-BPE` (9 Swahili/Kinyarwanda/merged variants) | Tokenization | varies | — | `olaverse` |
| `Reranker` (2 sizes) | Reranking | 23 MB – 150M params | — | `olaverse[retrieval]` |
| `Embedder` | Sentence embeddings (ha/yo/ig) | ~120 MB | — | `olaverse[retrieval]` |
| `PrismUpscaler` (3 sizes) | Image super-resolution | ~25K params – small | — | `olaverse[vision]` |
| `PrismDenoiser` | Image denoising | Small U-Net | — | `olaverse[vision]` |
| `PrismSteganography` | Image steganography | Small U-Net | — | `olaverse[vision]` |

</div>

---

## Which model should I use?

### Diacritization

| Need | Model |
|---|---|
| Fast Yoruba, CPU-only | `diacnet-yor-viterbi` |
| Highest Yoruba accuracy | `diacnet-yor-x` |
| Igbo | `diacnet-ig` |
| 10 languages, one model | `diacnet-1.0` |
| Automatic language routing | `Diacritizer(model="auto")` |

### Language detection

| Need | Model |
|---|---|
| Nigerian languages + English, zero GPU | `LIDLite5` |
| Nigerian languages + English, best accuracy | `LIDNeural5` |
| 25 languages, CPU-only | `LIDLite25` |
| 25 languages, best short-text accuracy | `LIDNeural25` |
| Nigerian languages only (input never contains English) | `LIDNeural5_1` |

### LLMs

| Need | Model |
|---|---|
| Fast everyday use, single consumer GPU | `MIST(size="8b")` |
| Structured, detailed output | `MIST(size="70b")` |
| Deepest reasoning | `MIST(size="140b")` |
| Step-by-step visible reasoning | `MIST(size="thinking")` |
| Contract analysis | `LegalPeace` |

---

## API Conventions

Olaverse model classes follow a consistent verb pattern — once you know one model, you know them all:

| Method | Used by | Meaning |
|---|---|---|
| `predict()` / `predict_proba()` | LID models | Classify input, optionally with probabilities |
| `predict_batch()` | LID models | Batched classification in one forward pass |
| `restore()` | `Diacritizer` | Restore diacritics to plain text |
| `encode()` / `decode()` | `Tokenizer`, `Embedder` | Text ↔ token IDs / embedding vectors |
| `rank()` / `score()` | `Reranker` | Order or score (query, passage) pairs |
| `normalize()` | `TTSNormalizer`, `NaijaNormalizer` | Expand numbers/abbreviations for speech |
| `generate()` / `chat()` | `MIST`, `LegalPeace` | LLM completion / multi-turn chat |
| `load()` | neural models | Explicit one-time weight download (lazy elsewhere) |

Small CPU models (e.g. `LIDLite5`, Viterbi diacritizers) load instantly at construction; neural models expose an explicit `load()` so the download happens where you expect it.
