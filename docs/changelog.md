# Changelog

All notable changes to the Olaverse SDK are documented here.

---

## Unreleased

### New Features

#### `LIDLite608` and `LIDNeural608` — language identification for 608 languages

```python
from olaverse import LIDLite608, LIDNeural608

LIDLite608().predict("Ẹ kú àárọ̀, ṣé dáadáa ni?")                 # → 'yor_Latn'  (fastText, CPU, 37 MB)
LIDNeural608(mode="traffic").predict_batch(["Good morning", "Habari za asubuhi"])
# → ['eng_Latn', 'swh_Latn']                                       (mmBERT, 140M)
```

- 608 languages across 36 scripts plus a `zxx_Zxxx` noise class, African-first. Labels are ISO 639-3 plus ISO 15924 script (`yor_Latn`); the 5- and 25-language classes keep returning bare codes.
- **Two modes** (`mode=`): `"coverage"` (default; every language equally likely, for corpus building and low-resource mining) and `"traffic"` (scores shifted by each language's real-world frequency via the models' `priors.json`, for user input and routing).
- `LIDLite608` needs `olaverse[lid]`; `LIDNeural608` needs `olaverse[deeplearning]` and, per its model card, `transformers>=5.14` (an older version fails with an error that says so). `LIDNeural608` takes `device=` and loads in bfloat16 on CUDA; it reads up to 512 tokens.
- Input is collapsed to one line, as both models expect; empty text raises `ValueError`.
- `LIDLite608` works with NumPy 2, where fasttext-wheel's own `predict()` fails (`Unable to avoid copy`). `LIDLite25` still calls that method and is unchanged.
- The shared transformer base class gained small hooks (`max_length`, `device`, probability and text-preparation hooks). Their defaults leave `LIDNeural5`, `LIDNeural5_1` and `LIDNeural25` behaving exactly as before.

### Fixed

- **A dropped download no longer leaves a broken file in the cache.** `get_model_path()` used to open the final cache path before downloading, so a connection that died mid-transfer left an empty or partial file there; the next call found it, returned it, and loading failed with `EOFError: Ran out of input`. Downloads now stream in 1 MB chunks into `<file>.part`, are retried up to 5 times after a drop (each retry resumes with a `Range` request from the bytes already on disk), are checked against the server's `Content-Length`, and are moved into place with `os.replace()` only when complete. An unfinished `.part` file is resumed by the next call, and a server that ignores `Range` just restarts the file. A **0-byte file counts as missing**, in the cache, in `OLAVERSE_MODELS_DIR` and in the bundled models folder, so a cache already damaged by an earlier release repairs itself. Reads that stall for 60 seconds are abandoned and retried instead of hanging.
- **A failed ONNX download is no longer reported as "No ONNX export found".** The diactag loader's optional lookups swallowed every error and returned "not there". Only a 404 means that now; a dropped connection, a 401 on a private repository or a 5xx is raised with its original message. This also covers `calibration.json`, where a failed download used to fall back silently to temperature 1.0.
- **`LIDLite25` works with NumPy 2.** fasttext-wheel's `predict()` fails there with `Unable to avoid copy while creating an array`; `LIDLite25` now uses the same fallback as `LIDLite608`. Any other error from fastText is raised as before.

### Changed

- `get_model_path()` raises `ModelNotFoundError` for a 404 and `ModelDownloadError` for any other download failure, both with a `status` attribute. They subclass `RuntimeError`, so existing `except RuntimeError` handlers keep working, and the message is the same as before.

---

## v0.4.0

**Released**: 2026-10-06

Arabic, and three new models: `diactag-2.0`, `diacnet-2.0` and `diacnet-mini-2.0`. Everything that worked on 0.3.1 — `diactag-1.0`, `diacnet-1.0`/`1.1` and the small Yoruba/Igbo models — behaves exactly as before.

### New Features

#### `diactag-2.0` — the per-character tagger, now with Arabic

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="diactag-2.0", lang="ar")          # "ar" or "ara"
d.restore("ذهب الطالب إلى المدرسة في الصباح")
Diacritizer(model="diactag-2.0", lang="ar", case_endings=False)
```

- 11 languages: the original ten plus Arabic (`ara`/`ar`; the LID head detects it too). Label space spec 2.0.0 — 986 characters, 18 shape classes, 15 tone classes. Harakat, tanwīn and sukūn are tone; shadda and dagger alif are shape; hamza letters (أ إ آ ؤ ئ) are spelling and are never added or removed.
- **`case_endings=`** (constructor and per call, default `True`). `False` drops the vowel / tanwīn / sukūn on each word's last Arabic letter, keeping shadda; a mark you typed yourself is left alone. No effect on other languages. On `diactag-1.0`, which has no Arabic, it raises.
- **Spec-gated decoding.** The SDK now ships two vendored copies of the decoding code, `olaverse.nlp._diactag` (spec 1.2.0, untouched) and `olaverse.nlp._diactag2` (spec 2.0.0, ported from the code in the model repo). Which one decodes a checkpoint is decided by the `spec_version` in its `labels.json`, so `diactag-1.0` keeps running on the code it was built with. The exact-match check in `LabelSpace.load` is unchanged: a 1.1.0 or 2.1.0 file is rejected, and an unknown major version fails with an error naming the specs involved.
- `DiacTagDecoder(ckpt=None)` now picks the release's own checkpoint (`ckpt_final.pt` for 2.0; `ckpt_120000.pt` for 1.0, as before).

#### `diacnet-2.0` and `diacnet-mini-2.0` — ByT5 text-to-text, 11 languages

- Prompt format `<tag> [g: word=meaning] text`. Tags: `yor ibo hau vie pol tur por spa fra ita ara ara-nocase auto`; `lang=None` sends `<auto>`. ISO-639-1 codes are accepted.
- **Meaning hints** — `restore(text, hints={"ranh": "free (time)"})` (or a list of `"word=meaning"` strings).
- **Output alignment, on by default (`aligned=True`).** Your letters are kept and only the model's marks are taken, so the output strips back to your input. `aligned=False` returns the raw generation (which can also repair typos, and may change a letter). Alignment protects letters, not marks: a mark you typed can still be replaced by the model's choice. Also available standalone as `olaverse.nlp.align_marks`.
- Text is chunked at ~300 characters on spaces, decoded greedily (`num_beams=1`, `max_new_tokens = 2 × input tokens + 16`), NFC-normalised and rejoined with single spaces.
- **Batching** — `Diacritizer.restore_batch(texts, ...)` pools the chunks of many texts, sorts them by length and decodes them in batches padded to the longest. `device=` (`"cpu"`, `"cuda"`, `"mps"`, `"auto"`) and `batch_size=`; weights load in bfloat16 on CUDA and float32 elsewhere.
- New public helpers in `olaverse.nlp.diacnet_utils`: `align`, `strip_marks`, `chunk_text`, `build_hint`, `resolve_tag`.

#### Also

- `MISTQuestionGenerator.build_messages(passage, n, language)` — public, static (no model load) helper returning the exact system + user chat messages `generate` sends. Use it to run the teacher prompt against your own endpoint without importing private names.
- `Diacritizer.restore_batch` works on every model; on the non-batching ones it restores each text in turn.

### Compatibility

- `diacnet-1.0`/`1.1` keep their prompt format, sentence splitting and raw output. They accept `aligned=True` as an opt-in; it is `False` by default, so existing output does not change.
- `Diacritizer.restore(...)` gained the keyword arguments `case_endings`, `aligned` and `hints`, appended after the existing ones. Passing one to a model that does not support it raises `ValueError`.
- `DiacTagDecoder.normalize_language` errors now name the model they came from instead of always saying `diactag-1.0`.

---

## v0.3.1

**Released**: 2026-08-04

Documentation only — no code changes.

The `diactag-1.0` docs presented lexicon reranking as a straightforward feature. Measured on diacbench it cuts non-word outputs by **27%** but raises Yorùbá DER by **15%** (0.0836 → 0.0961), which the model card documented and the SDK docs did not.

The cause is the density gating that makes the model good in the first place: restricting the lexicon to well-marked text shrank the Yorùbá vocabulary from 86k forms to **18,436** — the harshest cut of any language, because 89% of the Yorùbá corpus was under-marked. So "not in the lexicon" often means "rare or inflected word we didn't keep" rather than "wrong spelling", and correct output gets overwritten. The same gate left Igbo at 62,186 forms and Italian at 44,977, so the effect is far weaker there.

`use_lexicon` was already off by default; this release documents why you would leave it that way. The docstrings are corrected too, so `help(Diacritizer)` matches the docs site — that is the part which needed a release, since the docs site rebuilds from `main` but docstrings ship inside the wheel.

---

## v0.3.0

**Released**: 2026-08-04

### New Features

#### `diactag-1.0` — diacritization that cannot corrupt your text

A per-character tagger rather than a seq2seq model. It copies every base character through and only predicts which diacritics that character carries, so `strip(output) == strip(input)` holds **by construction** — for a trained model, an untrained one, or the int8 export. The SDK asserts it on every call.

This is a real failure the previous line had: on Hausa only 94.7% of `diacnet-1.1` outputs still stripped back to their input. The other 5.3% weren't mis-accented, they were different text — words dropped, clauses rewritten, punctuation invented.

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="diactag-1.0", lang="yo")
d.restore("se eranko naa si gbo o?")
# → 'ṣé ẹranko náà sì gbọ́ ọ?'
```

37.6M parameters against 580M, and it beats `diacnet-1.1` on 9 of 10 languages — Yoruba 0.2006 → **0.0836** DER, Hausa 0.0593 → **0.0041**.

**Built-in language detection.** Omit `lang=` and the model's own LID head decides, at a cost of ~0.0001 DER.

```python
d = Diacritizer(model="diactag-1.0")
d.restore("Co ay rat dam dang")           # → 'Cô ấy rất đảm đang'
d.detect_language("Lodz jest piekna")     # → ('pol', 0.9999)
```

**Per-character confidence and abstention.** Characters below `min_confidence` are left exactly as the caller typed them. At 0.9, ~97% of characters are restored at 99.6% accuracy and the rest are flagged. The threshold is also a per-call argument, so one loaded model can serve a CMS pre-fill and a legal pipeline at different points on the same curve.

```python
d = Diacritizer(model="diactag-1.0", lang="yor", min_confidence=0.9)

text, details = d.restore("se eranko naa", return_details=True)
review = [c for c in details if c.confidence < 0.9]
```

**int8 ONNX backend** — 3x faster and 4x smaller (38 MB, 244 chars/s on one CPU core) for +0.03pp DER. Compliance stays 1.0000 under quantisation, because the guarantee is architectural rather than a property of numeric precision.

```python
d = Diacritizer(model="diactag-1.0", lang="yor", onnx=True)   # pip install olaverse[onnx]
```

Language auto-detection works on the ONNX backend too, matching the PyTorch head to four decimal places. This needed a re-export: the previous ONNX graph carried the SHAPE and TONE heads only, and `lang_known` was hardcoded inside diactag's `ExportWrapper` — which matters, because the language embedding is additive at every position and leaks into the mean-pooled state the LID head reads, so a head exported under "language is known" just echoes the caller's own guess (Polish text with `lang="yor"` comes back `yor` at 0.9477). The published export now takes `lang_known` as a graph input and emits `lid_logits`.

The SDK reads that capability off the graph rather than assuming it, so a pinned older revision still loads — it just requires an explicit `lang=` and raises without one, instead of silently resolving every input to Yoruba.

Documents are handled natively with overlapping sliding windows — no sentence splitting needed. URLs, emails, `@handles` and `CONSTANT_NAMES` pass through untouched, and marks already present in the input are never silently deleted.

Full guide: **[DiacTag →](models/diactag.md)**

#### `diacnet-1.1`

The same ByT5 architecture retrained on a much larger web-sourced corpus. A large improvement on Vietnamese (0.1264 → 0.0460 DER), Turkish (0.0447 → 0.0068), Polish (0.0357 → 0.0058) and Italian (0.0015 → 0.0002); a **regression on Yoruba** (0.1554 → 0.2006), because the larger corpus is mostly under-tone-marked and the model learned to omit marks too. It does not supersede `diacnet-1.0` — pick per language, or use `diactag-1.0`, which fixes the regression outright.

```python
d = Diacritizer(model="diacnet-1.1", lang="vi")
d.restore("Toi khong biet tieng Viet")   # → 'Tôi không biết tiếng Việt'
```

### Fixes

#### Private and gated Hugging Face repositories now authenticate

The model downloader only read `HF_TOKEN` from the environment, so anyone who had authenticated the normal way — `huggingface-cli login`, which writes a token file — hit a bare `401` on a private repo. It now falls back to the CLI token store (`HF_HOME/token`, `~/.cache/huggingface/token`) and also accepts `HUGGING_FACE_HUB_TOKEN`.

### Notes

`Diacritizer.restore()` gained `lang=`, `min_confidence=` and `return_details=` keyword arguments. Passing a diactag-only argument to another model raises rather than being silently ignored, and `lang=` is now a per-call override for the multilingual models. Existing single-argument calls are unaffected.

---

## v0.2.1

**Released**: 2026-07-26

Documentation only — no code changes. The README shipped in v0.2.0 predated the new models, so the PyPI project page didn't mention `MISTTitleGenerator` or `MISTQuestionGenerator`. PyPI renders the README from the uploaded distribution and won't let a published version be edited, so refreshing that page needs a release.

---

## v0.2.0

**Released**: 2026-07-26

### New Features

#### `MISTTitleGenerator` and `MISTQuestionGenerator`

The two task-specific MIST models now have SDK classes instead of requiring raw `transformers` code.

```python
from olaverse import MISTTitleGenerator, MISTQuestionGenerator

MISTTitleGenerator().generate("My laptop keeps freezing when I open too many tabs, why?")
# → 'Laptop Freezing Impact'

MISTQuestionGenerator().generate(passage, n=3, language="fra")
```

`MISTQuestionGenerator` uses the teacher prompt the model was distilled with, sizes the JSON skeleton to `n`, and accepts ISO 639-3, ISO 639-1, or English language names. `language=` names the language the passage is written in — this is same-language question generation, not translation.

### Fixes

#### `diacnet-1.0` no longer truncates long text

The model was trained on sentence-length input (median 58 bytes) and the SDK fed whole paragraphs through in one pass, silently losing data — a 358-character French paragraph came back at 235 characters with its last two sentences dropped. Multi-sentence input is now segmented, restored a sentence at a time, and rejoined.

```python
Diacritizer(model="diacnet-1.0", lang="fr")                         # segments (default)
Diacritizer(model="diacnet-1.0", lang="fr", split_sentences=False)  # one pass
Diacritizer(model="diacnet-1.0", lang="fr", splitter=my_splitter)   # your own
```

### Documentation

Model claims were re-run against the real checkpoints and corrected where they didn't hold: `PrismDenoiser` always returns 128x128 rather than matching input, `PrismSteganography` messages do not survive a JPEG save at any quality, and the multilingual OTK-BPE tokenizers silently drop emoji.

---

## v0.1.5

**Released**: 2026-07-16

### New Features

#### 25-language identification (`LIDLite25`, `LIDNeural25`)

Extends language detection well beyond the original 5 Nigerian languages to 25 languages spanning Africa, Europe, and Asia. Each comes in `"passages"` (long-form) and `"questions"` (short-form) variants.

```python
from olaverse import LIDLite25, LIDNeural25

lite = LIDLite25(variant="questions")       # fastText, CPU-only
lite.predict("What causes ocean tides?")    # → 'eng'

neural = LIDNeural25(variant="questions")   # XLM-RoBERTa
neural.load()
neural.predict_proba("What causes ocean tides?")
```

#### `LIDNeural5_1` — compact Nigerian-only classifier

A ~31M parameter classifier built on the new `mist-encoder-base-ng` encoder, covering Hausa/Yoruba/Igbo/Nigerian Pidgin with no English fallback class.

```python
from olaverse import LIDNeural5_1

detector = LIDNeural5_1()
detector.predict("Ina kwana?")   # → 'Hausa'
```

#### `diacnet-1.0` — multilingual diacritization

A single joint ByT5 model restores diacritics across 10 languages (Yoruba, Igbo, Hausa, Vietnamese, Polish, Turkish, Portuguese, Spanish, French, Italian) via a `lang=` argument — no separate per-language model needed.

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="diacnet-1.0", lang="fr")
d.restore("Le cafe est tres chaud.")   # → 'Le café est très chaud.'
```

#### OTK-BPE multilingual tokenizer family

Swahili, Kinyarwanda, and a merged French/Kinyarwanda/English/Swahili tokenizer, each at 50k/100k/150k vocab sizes, through the same `Tokenizer` class used for the Nigerian-language family.

```python
from olaverse import Tokenizer

tok = Tokenizer("sw-150k")
ids = tok.encode("Habari yako?")
```

#### New retrieval toolkit (`olaverse.nlp.retrieval`)

`Reranker` (cross-encoder, 150M/22.7M variants) and `Embedder` (cross-lingual Hausa/Yoruba/Igbo sentence embeddings) for RAG/search pipelines.

```python
from olaverse import Reranker, Embedder

reranker = Reranker(size="22.7m")
reranker.rank("who wrote hamlet", ["Hamlet is a tragedy by Shakespeare.", "Paris is in France."])

embedder = Embedder()
vecs = embedder.encode(["bawo ni", "sannu"])
```

#### Dataset access (`olaverse.data`)

`load_dataset`, `list_datasets`, and `dataset_info` give one-line access to every public olaverse dataset on Hugging Face — reranker training pairs, multilingual QG passages, and the DiacBench diacritization benchmark. Requires `pip install olaverse[data]`. Replaces the old placeholder `olaverse.data.loaders` module, which returned bundled sample rows instead of real data.

```python
from olaverse import load_dataset

pairs = load_dataset("reranker-general-en-llm-judged", split="train")
bench = load_dataset("diacbench", "yo", split="test")
```

#### New `olaverse.vision` module

`PrismUpscaler` (2x/4x/arbitrary-resolution super-resolution), `PrismDenoiser` (noise/blur/JPEG-artifact removal), and `PrismSteganography` (hide/recover short messages in images) — general-purpose image-to-image models, not African-language-specific.

```python
from olaverse import PrismUpscaler, PrismDenoiser, PrismSteganography

PrismUpscaler(size="2x").upscale("input.jpg").save("output.jpg")
PrismDenoiser().denoise("noisy.jpg").save("denoised.jpg")

steg = PrismSteganography()
stego = steg.hide("cover.jpg", "hi there")
steg.reveal(stego)   # → 'hi there'
```

### Changes

- **New extras**: `olaverse[lid]` (fastText for `LIDLite25`), `olaverse[retrieval]` (`sentence-transformers`), `olaverse[vision]` (`torch`, `torchvision`, `Pillow`), `olaverse[data]` (Hugging Face `datasets`).
- **Fix — `diacritize_yoruba`** previously special-cased the documentation example sentence and returned a hand-written answer for it instead of the model output. The special case is removed; all inputs now go through the Viterbi model, and docs show the model's real output.
- **`Diacritizer`** gained a `lang=` constructor argument, used only by `model="diacnet-1.0"`.
- **`Tokenizer`** now resolves multilingual (`sw-*`/`kin-*`/`merged-*`) variants against the `olaverse/otk-bpe` repo, alongside the existing Nigerian-language `olaverse/otk-bpe-50k` repo.
- **`LIDNeural5`** internals refactored onto a shared base class (`LIDNeural25`, `LIDNeural5_1` reuse the same loading/inference logic) — no change to `LIDNeural5`'s public API.

### Install

```bash
pip install olaverse                # core NLP (no GPU required)
pip install olaverse[deeplearning]  # + LIDNeural5/25/51, diacnet-1.0, MIST local
pip install olaverse[lid]           # + LIDLite25 (fastText)
pip install olaverse[retrieval]     # + Reranker, Embedder
pip install olaverse[vision]        # + PrismUpscaler, PrismDenoiser, PrismSteganography
pip install olaverse[hosted]        # + MIST via Featherless/Modal
pip install olaverse[legal]         # + LegalPeace
pip install olaverse[data]          # + load_dataset (olaverse datasets on HF)
```

---

## v0.1.4

**Released**: 2026-06-15

### New Features

#### MIST Model Family (`olaverse.llm.MIST`)

Unified interface for all MIST variants — correct stop tokens, verified sampling defaults, and a local/hosted endpoint switch in one class.

```python
from olaverse import MIST

# Local (transformers)
model = MIST(size="8b")
model.load()
print(model.generate("What makes Yoruba a tonal language?"))

# Hosted (Featherless, Modal, or any OpenAI-compatible endpoint)
model = MIST(size="70b", endpoint="featherless", api_key="...")
print(model.generate("Write a Python retry decorator."))
```

Supported variants: `"8b"` / `"mini"`, `"70b"`, `"140b"`, `"140b-4bit"`, `"thinking"`.

#### Batch inference — `LIDNeural5.predict_batch` / `predict_proba_batch`

Single batched forward pass instead of per-text loops — significantly faster for dataset processing.

```python
detector = LIDNeural5()
detector.load()

langs = detector.predict_batch(["Bawo ni?", "Kedu?", "How far?"])
# → ['yor', 'ibo', 'pcm']

probs = detector.predict_proba_batch(["Bawo ni?", "Kedu?"])
# → [{'yor': 0.99, ...}, {'ibo': 0.98, ...}]
```

#### Auto-routing Diacritizer (`model="auto"`)

Detects language automatically via LIDLite5 and routes to the correct diacritizer — no need to specify the language.

```python
from olaverse.nlp import Diacritizer

d = Diacritizer(model="auto")
d.restore("Ojo lo si oja lana")   # detected: Yoruba → 'Òjó lọ sí ọjà lana'
d.restore("Kedu ka i mere")       # detected: Igbo   → 'Kedụ ka ị mere'
```

#### Stopwords (`olaverse.nlp.stopwords`)

Linguistic stopword sets for all 4 Nigerian languages plus convenience utilities.

```python
from olaverse import YORUBA_STOPWORDS, get_stopwords, filter_stopwords

# Direct set access
"ni" in YORUBA_STOPWORDS        # → True

# By language code
sw = get_stopwords("pcm")       # → PIDGIN_STOPWORDS

# Filter a token list
filter_stopwords(["bawo", "ni", "Ade", "dara"], "yor")
# → ['Ade', 'dara']
```

#### NaijaNormalizer (`olaverse.nlp.NaijaNormalizer`)

Pidgin-specific TTS normalizer extending `TTSNormalizer`. Adds informal spelling normalization (e.g. `"2moro"` → `"tomorrow"`, `"nd"` → `"and"`) on top of the standard abbreviation + number pipeline.

```python
from olaverse import NaijaNormalizer

norm = NaijaNormalizer()
norm.normalize("Oga, e don finish. Call am 2moro pls.")
# → 'Oga, e don finish. Call am tomorrow please.'
```

#### `MIST` retry logic for hosted inference

Automatic retry on capacity/overload errors with configurable attempts and delay.

```python
model = MIST(
    size="70b",
    endpoint="featherless",
    api_key="...",
    max_retries=3,      # default: 3
    retry_delay=5.0,    # seconds; each attempt waits delay × attempt_number
)
```

### Changes

- **`LIDNeural5` moved to `olaverse.nlp`** — its correct home alongside `LIDLite5`. `from olaverse.llm import LIDNeural5` continues to work (backward-compat re-export in `llm/detector.py`).
- **`LIDNeural5` now exported from `olaverse.nlp`** — `from olaverse.nlp import LIDNeural5` is now the canonical import path.
- **Speech demoted to Experimental** — `TTSPipeline`, `BaseAcousticModel`, `BaseVocoder` emit `ExperimentalWarning` on use. No trained acoustic model or vocoder exists yet. The diacritization and normalization steps remain production-ready.
- **`NaijaNormalizer` added to `TTSNormalizer` Pidgin support** — `TTSNormalizer(lang="pcm")` now has a populated abbreviation and digit table (was previously empty).
- **New `olaverse[hosted]` extra** — `pip install olaverse[hosted]` installs `openai>=1.0.0` for MIST hosted inference.
- **`ExperimentalWarning`** exported from `olaverse` and `olaverse.speech` for easy suppression.

### Install

```bash
pip install olaverse           # core NLP (no GPU required)
pip install olaverse[deeplearning]  # + LIDNeural5, MIST local
pip install olaverse[hosted]        # + MIST via Featherless/Modal
pip install olaverse[legal]         # + LegalPeace
```

---

## v0.1.3

**Released**: 2026-05-01 *(approximate)*

### Features

- **`LegalPeace`** — Fine-tuned Mistral-7B-v0.3 for contract analysis and legal reasoning. 4-bit quantized inference via unsloth. Achieves 10.3% faster inference and 32.6% faster contract analysis vs. base Mistral-7B.
- **`LIDNeural5`** *(initially in `olaverse.llm`)* — XLM-RoBERTa sequence classifier fine-tuned on 5 Nigerian languages, 98.96% macro-F1. Available via `pip install olaverse[deeplearning]`.
- **`LIDLite5`** — TF-IDF + Logistic Regression language detector. Zero GPU, 1.1 MB model file, 0.014 ms/sentence, 98.12% macro-F1. Available in core install.
- **`Diacritizer`** with 5 backends — Viterbi, KNN, dot-below KNN, BiLSTM, and XLM-RoBERTa transformer for Yoruba; KNN for Igbo.
- **`Tokenizer`** — OTK-BPE-50k family: Yoruba (63% fewer tokens vs GPT-4), Igbo, Hausa, Pidgin, and unified Naija.
- **`TTSNormalizer`** — Abbreviation and number expansion for Yoruba and Igbo TTS.
- **`mask_pii`** / **`clean_text`** — PII masking (emails, phones, credit cards, SSNs) and general text cleaning.
- **`TTSPipeline`** + **`BaseAcousticModel`** + **`BaseVocoder`** — TTS pipeline architecture (NLP front-end only; acoustic synthesis in development).
- **`olaverse.utils`** — Nigerian currency formatting, continent codes, `.wav` audio I/O.

---

## Roadmap

!!! note "Coming in future releases"
    - **Acoustic model + vocoder** for end-to-end Yoruba TTS (completes the speech pipeline)
    - **`Diacritizer` for Hausa** (`diacnet-ha`)
    - **`LIDNeural5_1` v5.2** — adds an English/"other" class, removing the confident-mislabelling failure mode of v5.1
    - **`MIST` embedding endpoint** for semantic search over Nigerian language content
    - **ASR (Automatic Speech Recognition)** for Nigerian languages
    - **Hausa / Pidgin TTS normalizer expansion**

!!! success "Done — was on the roadmap"
    - ~~`LIDNeural5` expanded to 10+ languages~~ — shipped in v0.1.5 as `LIDLite25`/`LIDNeural25` (25 languages total: Afrikaans, Amharic, German, English, French, Hausa, Hindi, Igbo, Indonesian, Italian, Japanese, Korean, Dutch, Polish, Portuguese, Russian, Shona, Somali, Spanish, Swahili, Turkish, Vietnamese, Xhosa, Yoruba, Zulu). Note: Efik, Tiv, and Nupe specifically are still not covered by any olaverse LID model.
