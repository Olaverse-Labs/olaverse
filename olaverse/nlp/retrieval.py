"""
Retrieval utilities — cross-encoder reranking and Nigerian-language sentence embeddings.

Requires: pip install olaverse[retrieval]
"""
from __future__ import annotations

_RERANKER_ALIASES = {
    "150m": "olaverse/mist-reranker-150m",
    "22.7m": "olaverse/mist-reranker-22.7M",
}


class Reranker:
    """
    Cross-encoder reranker for the second stage of a RAG / search pipeline.

    Scores (query, passage) pairs to re-sort the top-k candidates from a
    first-stage retriever (BM25 or a bi-encoder).

    Models (size=):
        "150m"   — mist-reranker-150m   (ModernBERT-base, English, best QA/fact accuracy)
        "22.7m"  — mist-reranker-22.7M  (MiniLM-L6, English, smaller/faster) [default]

    Requires: pip install olaverse[retrieval]

    Quick start:
        >>> reranker = Reranker(size="22.7m")
        >>> reranker.rank("who wrote hamlet", [
        ...     "Hamlet is a tragedy written by William Shakespeare.",
        ...     "The capital of France is Paris.",
        ... ])
        [(0, 0.98...), (1, 0.01...)]
    """

    def __init__(self, size: str = "22.7m"):
        size_lower = size.lower()
        self.model_name = _RERANKER_ALIASES.get(size_lower, size)
        self._model = None

    def load(self):
        """Download and load the reranker (runs once; cached after first call)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' library is required to load Reranker. "
                "Install with: pip install olaverse[retrieval]"
            )

        self._model = CrossEncoder(self.model_name)

    def score(self, query: str, passages: list) -> list:
        """
        Score a query against a list of passages.

        Args:
            query: The search query.
            passages: List of candidate passage strings.

        Returns:
            list[float]: relevance scores, one per passage, same order as input.
        """
        if self._model is None:
            self.load()

        import torch

        raw = self._model.predict([(query, p) for p in passages], convert_to_tensor=True)
        if raw.dim() == 2:
            # 2-class logits (e.g. mist-reranker-22.7M) — relevance = P(relevant)
            scores = torch.softmax(raw, dim=-1)[:, 1]
        else:
            scores = raw
        return scores.tolist()

    def rank(self, query: str, passages: list) -> list:
        """
        Rank passages by relevance to the query, descending.

        Args:
            query: The search query.
            passages: List of candidate passage strings.

        Returns:
            list[tuple[int, float]]: (original_index, score) pairs, best-first.
        """
        scores = self.score(query, passages)
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)


class Embedder:
    """
    Cross-lingual sentence embeddings for Nigerian languages (Hausa, Yoruba, Igbo).

    Wraps olaverse/naija-embed-base — contrastively fine-tuned from
    olaverse/mist-encoder-base-ng on synthetic parallel pairs. Mean pooling,
    cosine similarity. Useful for cross-lingual retrieval, semantic search,
    clustering, and deduplication over Nigerian-language text.

    Note: does not cover Nigerian Pidgin (pcm) — the base model only supports ha/yo/ig.

    Requires: pip install olaverse[retrieval]

    Quick start:
        >>> embedder = Embedder()
        >>> vecs = embedder.encode(["bawo ni", "sannu"])
        >>> embedder.similarity(vecs[0], vecs[1])
    """

    def __init__(self, model_name: str = "olaverse/naija-embed-base"):
        self.model_name = model_name
        self._model = None

    def load(self):
        """Download and load the embedding model (runs once; cached after first call)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "The 'sentence-transformers' library is required to load Embedder. "
                "Install with: pip install olaverse[retrieval]"
            )

        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: str | list[str], **kwargs: object) -> "numpy.ndarray":
        """
        Encode a string or list of strings into embedding vector(s).

        Args:
            texts: A string, or list of strings.
            **kwargs: Passed through to SentenceTransformer.encode().

        Returns:
            numpy.ndarray: embedding vector(s).
        """
        if self._model is None:
            self.load()
        return self._model.encode(texts, **kwargs)

    def similarity(self, a, b) -> float:
        """Cosine similarity between two embedding vectors."""
        import numpy as np

        a, b = np.asarray(a), np.asarray(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
