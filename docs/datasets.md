# Datasets

The Olaverse SDK gives you direct access to every public [olaverse dataset on Hugging Face](https://huggingface.co/olaverse) — the training and evaluation data behind the MIST rerankers, `mist-qg-1.5b`, `lid-lite-25`, and `diacnet-1.0`.

Dataset loading is a thin wrapper around the Hugging Face `datasets` library, with short names, config validation, and helpful errors.

## Install

```bash
pip install olaverse[data]
```

## Quick Start

```python
from olaverse import load_dataset, list_datasets, dataset_info

# Discover what's available
list_datasets()
# → ['reranker-general-en-llm-judged', 'marco-style-pairs-multi',
#    'qg-passages-multi', 'reranker-triples-multi',
#    'qg-eval-multi-fresh', 'diacbench']

# Inspect a dataset before loading
dataset_info("diacbench")
# → {'repo_id': 'olaverse/diacbench', 'configs': ['es', 'fr', 'ha', ...],
#    'splits': ['test'], ...}

# Load — returns a standard Hugging Face Dataset
ds = load_dataset("diacbench", "yo", split="test")
ds[0]
# → {'input': 'Titi di igba ti o maa fi ko eru re lo patapata, ...',
#    'reference': 'Títí di ìgbà tí ó máa fi kó ẹrù rẹ̀ lọ pátápátá, ...'}
```

`load_dataset` accepts either the short name (`"diacbench"`) or the full repo ID (`"olaverse/diacbench"`), and passes extra keyword arguments straight through to `datasets.load_dataset` — e.g. `streaming=True` for large datasets.

## Available Datasets

| Dataset | Task | Languages | Configs | Splits |
|---|---|---|---|---|
| [`reranker-general-en-llm-judged`](https://huggingface.co/datasets/olaverse/reranker-general-en-llm-judged) | Reranker / retriever training (LLM-judged graded relevance) | English | `pairs-graded` (default), `triplets` | `train`, `test` |
| [`marco-style-pairs-multi`](https://huggingface.co/datasets/olaverse/marco-style-pairs-multi) | (query, positive) pairs for embedding training | 25 languages | — | `train` |
| [`qg-passages-multi`](https://huggingface.co/datasets/olaverse/qg-passages-multi) | Passages + search-style questions (behind `mist-qg-1.5b`) | 25 languages | — | `train` |
| [`reranker-triples-multi`](https://huggingface.co/datasets/olaverse/reranker-triples-multi) | Reranker triples with hard negatives | 25 languages | — | `train` |
| [`qg-eval-multi-fresh`](https://huggingface.co/datasets/olaverse/qg-eval-multi-fresh) | Held-out question-generation eval (625 passages) | 25 languages | — | `train` |
| [`diacbench`](https://huggingface.co/datasets/olaverse/diacbench) | DiacBench — diacritization benchmark (~1,000 pairs/language) | 10 languages | one per language: `es fr ha ig it pl pt tr vi yo` | `test` |

## Examples

### Train a reranker — graded pairs or triplets

```python
from olaverse import load_dataset

# 844k LLM-judged (query, passage, grade) pairs — cross-encoder training
pairs = load_dataset("reranker-general-en-llm-judged", "pairs-graded", split="train")

# 82k (query, positive, negative_1..5) triplets — bi-encoder / ColBERT training
triplets = load_dataset("reranker-general-en-llm-judged", "triplets", split="train")
```

### Benchmark a diacritizer on DiacBench

```python
from olaverse import load_dataset
from olaverse.nlp import Diacritizer

bench = load_dataset("diacbench", "yo", split="test")
d = Diacritizer(model="diacnet-yor-viterbi")

restored = d.restore(bench[0]["input"])
reference = bench[0]["reference"]
```

### Stream a large multilingual dataset

```python
from olaverse import load_dataset

stream = load_dataset("qg-passages-multi", split="train", streaming=True)
for row in stream:
    print(row)
    break
```

## Errors you might see

- **`ValueError: Dataset 'diacbench' requires a config`** — multi-config datasets like DiacBench need an explicit config, e.g. `load_dataset("diacbench", "yo")`.
- **`ImportError: The 'datasets' library is required`** — install the extra: `pip install olaverse[data]`.

---

## API Reference

::: olaverse.data.load_dataset

::: olaverse.data.list_datasets

::: olaverse.data.dataset_info
