"""
Access to the public Olaverse datasets on Hugging Face.

Thin wrapper around the Hugging Face ``datasets`` library with a registry of
every olaverse dataset — short names, available configs/splits, and helpful
errors when a config is required.

Requires: pip install olaverse[data]
"""
from __future__ import annotations

# Registry of all public olaverse datasets on Hugging Face.
# configs: None means the dataset has a single default config.
DATASETS = {
    "reranker-general-en-llm-judged": {
        "repo_id": "olaverse/reranker-general-en-llm-judged",
        "description": (
            "Graded English query-passage relevance (LLM-judged, 0-3) for training "
            "rerankers and retrievers. Trained mist-reranker-150m / mist-reranker-22.7M."
        ),
        "configs": ["pairs-graded", "triplets"],
        "default_config": "pairs-graded",
        "splits": ["train", "test"],
    },
    "marco-style-pairs-multi": {
        "repo_id": "olaverse/marco-style-pairs-multi",
        "description": (
            "Multilingual (query, positive passage) pairs across 25 languages "
            "for bi-encoder / embedding training."
        ),
        "configs": None,
        "default_config": None,
        "splits": ["train"],
    },
    "qg-passages-multi": {
        "repo_id": "olaverse/qg-passages-multi",
        "description": (
            "Passages paired with search-style questions across 25 languages — "
            "the training set behind mist-qg-1.5b, lid-lite-25, and diacnet-1.0."
        ),
        "configs": None,
        "default_config": None,
        "splits": ["train"],
    },
    "reranker-triples-multi": {
        "repo_id": "olaverse/reranker-triples-multi",
        "description": (
            "Multilingual reranker training triples (query, positive, hard negatives) "
            "across 25 languages."
        ),
        "configs": None,
        "default_config": None,
        "splits": ["train"],
    },
    "qg-eval-multi-fresh": {
        "repo_id": "olaverse/qg-eval-multi-fresh",
        "description": (
            "Held-out multilingual question-generation eval set (625 passages, "
            "never seen in training) used to benchmark mist-qg-1.5b."
        ),
        "configs": None,
        "default_config": None,
        "splits": ["train"],
    },
    "diacbench": {
        "repo_id": "olaverse/diacbench",
        "description": (
            "DiacBench — diacritization benchmark: ~1,000 diacritic-stripped/reference "
            "sentence pairs per language, one config per language."
        ),
        "configs": ["es", "fr", "ha", "ig", "it", "pl", "pt", "tr", "vi", "yo"],
        "default_config": None,  # a language config must be chosen explicitly
        "splits": ["test"],
    },
}


def _resolve_name(name: str) -> str:
    """Accept either a short name ('diacbench') or a full repo ID ('olaverse/diacbench')."""
    key = name.strip()
    if key.startswith("olaverse/"):
        key = key[len("olaverse/"):]
    if key not in DATASETS:
        raise ValueError(
            f"Unknown olaverse dataset {name!r}. "
            f"Available: {list(DATASETS)}"
        )
    return key


def list_datasets() -> list:
    """
    List the short names of all public olaverse datasets.

    Returns:
        list[str]: dataset names usable with load_dataset()/dataset_info().

    Quick start:
        >>> from olaverse import list_datasets
        >>> list_datasets()
        ['reranker-general-en-llm-judged', 'marco-style-pairs-multi', ...]
    """
    return list(DATASETS)


def dataset_info(name: str) -> dict:
    """
    Return registry metadata for one dataset: Hugging Face repo ID,
    description, available configs, and splits.

    Args:
        name: Short name (e.g. "diacbench") or full repo ID ("olaverse/diacbench").

    Returns:
        dict: {'repo_id', 'description', 'configs', 'default_config', 'splits'}
    """
    return dict(DATASETS[_resolve_name(name)])


def load_dataset(name: str, config: str = None, split: str = None, **kwargs: object) -> object:
    """
    Load an olaverse dataset from Hugging Face.

    Thin wrapper around ``datasets.load_dataset`` that resolves short names,
    validates configs, and gives actionable errors.

    Args:
        name: Short name (e.g. "diacbench") or full repo ID ("olaverse/diacbench").
        config: Config name for multi-config datasets
                (e.g. "yo" for diacbench, "triplets" for reranker-general-en-llm-judged).
        split: Optional split, e.g. "train" or "test". When omitted, returns a
               DatasetDict with every available split.
        **kwargs: Passed through to ``datasets.load_dataset``
                  (e.g. streaming=True).

    Returns:
        A ``datasets.Dataset`` when ``split`` is given, otherwise a
        ``datasets.DatasetDict`` with every available split.

    Requires: pip install olaverse[data]

    Quick start:
        >>> from olaverse import load_dataset
        >>> diacbench_yo = load_dataset("diacbench", "yo", split="test")
        >>> pairs = load_dataset("reranker-general-en-llm-judged", split="train")
        >>> qg = load_dataset("qg-passages-multi", split="train")
    """
    key = _resolve_name(name)
    info = DATASETS[key]

    if info["configs"]:
        if config is None:
            config = info["default_config"]
        if config is None:
            raise ValueError(
                f"Dataset '{key}' requires a config. "
                f"Available configs: {info['configs']}. "
                f"Example: load_dataset('{key}', '{info['configs'][0]}')"
            )
        if config not in info["configs"]:
            raise ValueError(
                f"Unknown config {config!r} for dataset '{key}'. "
                f"Available configs: {info['configs']}"
            )
    elif config is not None:
        raise ValueError(f"Dataset '{key}' has no configs, but config={config!r} was given.")

    try:
        import datasets as hf_datasets
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to load olaverse datasets. "
            "Install with: pip install olaverse[data]"
        )

    if config is not None:
        return hf_datasets.load_dataset(info["repo_id"], config, split=split, **kwargs)
    return hf_datasets.load_dataset(info["repo_id"], split=split, **kwargs)
