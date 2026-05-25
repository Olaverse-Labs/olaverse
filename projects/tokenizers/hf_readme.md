---
license: apache-2.0
library_name: tokenizers
tags:
- BPE
- tokenizer
- nigerian-languages
- yoruba
- igbo
- hausa
- pidgin
- naija
datasets:
- google/WaxalNLP
- wikimedia/wikipedia
- masakhane/masakhanews
---

# OTK-BPE-50k: Optimized BPE Tokenizers

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Olaverse-Labs/otk-bpe)

This repository hosts the **OTK-BPE-50k** family of production-grade Byte-Level BPE (BBPE) tokenizers designed for regional, custom, and multilingual language models.

It currently features models optimized for primary Nigerian languages: **Yoruba (`yo`)**, **Igbo (`ig`)**, **Hausa (`ha`)**, and **Nigerian Pidgin (`pcm`)**, along with a **Unified (Naija)** model. All models in this repository share a **50,000 vocabulary size (50k)** configuration. 

This repository will be expanded with 50k tokenizer models for other regional and low-resource languages in the future.

## Key Features
1. **Byte-Level BPE (BBPE)**: Maps raw UTF-8 bytes to printable characters, resulting in **0.00% Out-Of-Vocabulary (OOV)** / zero `[UNK]` tokens.
2. **Diacritic Preservation & Normalization**: Uses compiled `NFC` normalization inside the pre-tokenization chain to prevent tonal accents and combining diacritics from splitting into decomposed code points.
3. **Code-Mixed English Support**: Blends ~15% English Wikipedia articles into the training corpora, ensuring extremely low fertility rates when processing mixed sentences.
4. **Emoji & Symbol Merging**: Injects curated emoji vocab subsets during training to force BPE to merge them into single tokens rather than breaking them down into multiple byte pieces.
5. **Optimized Vocab Size (50,000)**: Restores the compression efficiency of monolingual tokenizers while preserving code-mixed English and emoji benefits.

---

## Tokenizer Models in this Repository

| Filename | Language | Target Language Code | Vocab Size |
| :--- | :--- | :--- | :--- |
| `otk-bpe-50k-yo.json` | Yoruba | `yo` | 50,000 |
| `otk-bpe-50k-ig.json` | Igbo | `ig` | 50,000 |
| `otk-bpe-50k-ha.json` | Hausa | `ha` | 50,000 |
| `otk-bpe-50k-pcm.json` | Nigerian Pidgin | `pcm` | 50,000 |
| `otk-bpe-50k-naija.json` | Unified (Naija Multilingual) | `naija` / `mul` | 50,000 |

---

## Performance Benchmarks
We evaluated the models against global multilingual baselines on the respective **MasakhaNEWS** test splits. 
*Fertility represents the average number of tokens produced per word (lower is better).*

### 1. Yoruba News Benchmarks (179,432 words)
| Tokenizer | Vocab Size | Total Tokens | Fertility | UNK Rate |
|---|---|---|---|---|
| **Olaverse Yoruba (BBPE)** | **50,000** | **223,615** | **1.246** (Best!) | **0.00%** |
| **Olaverse Unified (Naija)** | **50,000** | **232,628** | **1.296** | **0.00%** |
| GPT-4o (`o200k_base`) | 200,019 | 302,744 | 1.687 | 0.00% |
| AfroXLMR | 250,002 | 408,611 | 2.277 | 0.02% |
| GPT-4 (`cl100k_base`) | 100,277 | 455,352 | 2.538 | 0.00% |

### 2. Igbo News Benchmarks (141,600 words)
| Tokenizer | Vocab Size | Total Tokens | Fertility | UNK Rate |
|---|---|---|---|---|
| **Olaverse Igbo (BBPE)** | **50,000** | **196,819** | **1.390** (Best!) | **0.00%** |
| **Olaverse Unified (Naija)** | **50,000** | **200,576** | **1.416** | **0.00%** |
| GPT-4o (`o200k_base`) | 200,019 | 255,897 | 1.807 | 0.00% |
| GPT-4 (`cl100k_base`) | 100,277 | 370,053 | 2.613 | 0.00% |
| AfroXLMR | 250,002 | 363,965 | 2.570 | 0.00% |

### 3. Hausa News Benchmarks (304,201 words)
| Tokenizer | Vocab Size | Total Tokens | Fertility | UNK Rate |
|---|---|---|---|---|
| **Olaverse Hausa (BBPE)** | **50,000** | **369,035** | **1.213** (Best!) | **0.00%** |
| **Olaverse Unified (Naija)** | **50,000** | **374,380** | **1.231** | **0.00%** |
| GPT-4o (`o200k_base`) | 200,019 | 483,309 | 1.589 | 0.00% |
| AfroXLMR | 250,002 | 488,035 | 1.604 | 0.01% |
| GPT-4 (`cl100k_base`) | 100,277 | 625,640 | 2.057 | 0.00% |

### 4. Pidgin News Benchmarks (136,233 words)
| Tokenizer | Vocab Size | Total Tokens | Fertility | UNK Rate |
|---|---|---|---|---|
| **Olaverse Pidgin (BBPE)** | **50,000** | **166,186** | **1.220** (Best!) | **0.00%** |
| **Olaverse Unified (Naija)** | **50,000** | **170,174** | **1.249** | **0.00%** |
| GPT-4o (`o200k_base`) | 200,019 | 177,712 | 1.304 | 0.00% |
| GPT-4 (`cl100k_base`) | 100,277 | 184,129 | 1.352 | 0.00% |
| AfroXLMR | 250,002 | 190,921 | 1.401 | 0.00% |

---

## How to Use

### Installation
Make sure you have the required libraries installed:
```bash
pip install tokenizers transformers huggingface_hub
```

### Method A: Standard Transformers Loading (Recommended for LLM Training/Inference)
You can load the wrapped tokenizers directly from subfolders using Hugging Face's `AutoTokenizer`:

```python
from transformers import AutoTokenizer

# Load the custom Yoruba tokenizer
tokenizer = AutoTokenizer.from_pretrained("olaverse/otk-bpe-50k", subfolder="yo")

# Encode text
text = "Ẹ kú àbọ̀ ooo, ṣé dáadáa ni? 😂"
inputs = tokenizer(text)

print("Tokens:", tokenizer.tokenize(text))
print("IDs:", inputs["input_ids"])
print("Decoded:", tokenizer.decode(inputs["input_ids"]))
```

### Method B: Lightweight Raw BPE Loading (Using the `olaverse` library)
To avoid importing the heavy `transformers` package, use the lightweight, offline-first `olaverse` library:

```python
from olaverse import Tokenizer

# 1. Initialize tokenizer (it will automatically download and cache from the Hub on-demand)
tokenizer = Tokenizer("yo")  # Supports: "yo", "ig", "ha", "pcm", "naija"

# 2. Encode text
text = "Ẹ kú àbọ̀ ooo, ṣé dáadáa ni? 😂"
ids = tokenizer.encode(text)

print("IDs:", ids)
print("Decoded:", tokenizer.decode(ids))
```

---

## Datasets Used for Training
* **WaxalNLP**: Colloquial scripts for Yoruba (`yor_tts`), Igbo (`ibo_tts`), Hausa (`hau_tts`), and Pidgin (`pcm_tts`).
* **Wikipedia**: Monolingual encyclopedic corpora (`yo`, `ig`, `ha`, `pcm`).
* **MasakhaNEWS**: Nigerian language news articles.
* **Wikipedia (English)**: Clean English articles for code-mixed blending.
* **Curated Emoji Registry**: Common Nigerian social media emojis.
