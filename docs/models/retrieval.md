# Retrieval — Reranker & Embedder

**A two-piece toolkit for RAG and search pipelines: a cross-encoder reranker for precision, and cross-lingual Nigerian-language sentence embeddings for recall.**

```bash
pip install olaverse[retrieval]
```

```text
Query
  |
First-stage retrieval  (BM25 or Embedder)
  |
Top-k candidates
  |
Reranker  (cross-encoder rescoring)
  |
Final ranked results
```

---

## Reranker

Scores `(query, passage)` pairs to re-sort the top-k candidates from a first-stage retriever.

**Model Cards**: [olaverse/mist-reranker-150m](https://huggingface.co/olaverse/mist-reranker-150m) · [olaverse/mist-reranker-22.7M](https://huggingface.co/olaverse/mist-reranker-22.7M)

### Which size?

| `size=` | Params | Backbone | Best for |
|---|---|---|---|
| `"22.7m"` *(default)* | ~22.7M (23 MB) | MiniLM-L6 | Speed, CPU serving, low latency |
| `"150m"` | ~150M | ModernBERT-base | Best QA/fact accuracy |

### Usage

```python
from olaverse import Reranker

reranker = Reranker(size="22.7m")

reranker.rank("who wrote hamlet", [
    "Hamlet is a tragedy written by William Shakespeare.",
    "The capital of France is Paris.",
])
# → [(0, 0.915...), (1, 0.301...)]   # (original_index, score), best-first

reranker.score("who wrote hamlet", ["Hamlet is a tragedy by Shakespeare."])
# → [0.912...]
```

!!! note "English-only"
    Both reranker sizes are English-only. `Reranker` auto-handles their different output head shapes (single relevance score vs. 2-class logits).

---

## Embedder

Cross-lingual sentence embeddings for **Hausa, Yoruba, and Igbo** — contrastively fine-tuned from [`mist-encoder-base-ng`](https://huggingface.co/olaverse/mist-encoder-base-ng). A Hausa query can retrieve a Yoruba document.

**Model Card**: [olaverse/naija-embed-base](https://huggingface.co/olaverse/naija-embed-base)

```python
from olaverse import Embedder

embedder = Embedder()
vecs = embedder.encode(["bawo ni", "sannu"])
embedder.similarity(vecs[0], vecs[1])
```

!!! note "No Nigerian Pidgin support"
    The underlying translation model used for training only outputs Hausa/Yoruba/Igbo — Pidgin (`pcm`) is not covered.

---

## Training data

The datasets behind these models are public and loadable in one line:

```python
from olaverse import load_dataset

# 844k LLM-judged (query, passage, grade) pairs — cross-encoder training
pairs = load_dataset("reranker-general-en-llm-judged", "pairs-graded", split="train")

# 82k triplets with hard negatives — bi-encoder / ColBERT training
triplets = load_dataset("reranker-general-en-llm-judged", "triplets", split="train")
```

See **[Datasets →](../datasets.md)** for the full catalog.

---

## Applications

- ✅ **RAG pipelines** — rerank retrieved chunks before they hit the LLM context
- ✅ **Cross-lingual search** — query in Hausa, match documents in Yoruba or Igbo
- ✅ **Semantic deduplication** — cluster and dedupe multilingual corpora
- ✅ **FAQ / support matching** — map user questions to known answers

---

## API Reference

Full class reference: [NLP & Tokenization → Retrieval](../nlp.md#retrieval-new-in-v015)
