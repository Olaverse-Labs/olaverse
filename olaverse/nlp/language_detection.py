import os
import json
import math
import re
from olaverse.utils.downloader import get_model_path

_MODEL_CACHE = {}

def _load_model(model_name_or_path="lid-lite-5.json"):
    global _MODEL_CACHE
    
    resolved_path = model_name_or_path
    if not os.path.exists(resolved_path):
        try:
            resolved_path = get_model_path(model_name_or_path, repo_id="olaverse/lid-lite-5")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", model_name_or_path)
            
    if resolved_path in _MODEL_CACHE:
        return _MODEL_CACHE[resolved_path]
        
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"LIDLite5 model not found at: {resolved_path}")
        
    with open(resolved_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)
    _MODEL_CACHE[resolved_path] = model_data
    return model_data

class LIDLite5:
    """
    Lightweight, zero-dependency TF-IDF + Logistic Regression Language Detector for 5 languages:
    Yoruba ('yor'), Hausa ('hau'), Igbo ('ibo'), Pidgin ('pcm'), and English ('eng').
    """
    def __init__(self, model_path="lid-lite-5.json"):
        self.model_data = _load_model(model_path)
        self.classes = self.model_data["classes"]
        self.intercept = self.model_data["intercept"]
        self.features = self.model_data["features"]
        
    def _extract_features(self, text):
        text = text.lower().strip()
        words = re.findall(r'\b\w+\b', text)
        features = list(words)
        for i in range(len(words) - 1):
            features.append(f"{words[i]} {words[i+1]}")
        return features
        
    def predict_scores(self, text):
        if not text or not isinstance(text, str) or not text.strip():
            return {cls: 0.0 for cls in self.classes}
            
        features = self._extract_features(text)
        
        # Count term frequencies (TF)
        counts = {}
        for feat in features:
            counts[feat] = counts.get(feat, 0) + 1
            
        # Compute raw TF-IDF using sublinear scaling
        raw_tfidf = {}
        for feat, count in counts.items():
            if feat in self.features:
                tf = 1.0 + math.log(count)
                raw_tfidf[feat] = tf * self.features[feat]["idf"]
                
        if not raw_tfidf:
            # Fallback to intercepts if no features match
            return {cls: self.intercept[idx] for idx, cls in enumerate(self.classes)}
            
        # L2 normalization
        l2_norm = math.sqrt(sum(val ** 2 for val in raw_tfidf.values()))
        norm_tfidf = {feat: val / l2_norm for feat, val in raw_tfidf.items()}
        
        # Dot-product scoring
        scores = [0.0] * len(self.classes)
        for feat, val in norm_tfidf.items():
            weights = self.features[feat]["weights"]
            for idx in range(len(self.classes)):
                scores[idx] += val * weights[idx]
                
        for idx in range(len(self.classes)):
            scores[idx] += self.intercept[idx]
            
        return {cls: scores[idx] for idx, cls in enumerate(self.classes)}
        
    def predict(self, text):
        """
        Predict the language of the given text.
        Returns: 'yor', 'hau', 'ibo', 'pcm', or 'eng'.
        """
        scores = self.predict_scores(text)
        return max(scores, key=scores.get)
        
    def predict_proba(self, text):
        """
        Predict the language probabilities using softmax over logits.
        """
        scores = self.predict_scores(text)
        
        max_score = max(scores.values())
        exp_scores = {cls: math.exp(score - max_score) for cls, score in scores.items()}
        sum_exp = sum(exp_scores.values())
        
        return {cls: val / sum_exp for cls, val in exp_scores.items()}

def detect_language(text, model_path="lid-lite-5.json"):
    """
    Detect the language of the given text using LIDLite5.
    Returns: 'yor' (Yoruba), 'hau' (Hausa), 'ibo' (Igbo), 'pcm' (Pidgin), or 'eng' (English).
    """
    try:
        detector = LIDLite5(model_path)
    except (FileNotFoundError, KeyError):
        # Fallback to legacy filename for backward compatibility
        try:
            detector = LIDLite5("language_detector.json")
        except (FileNotFoundError, KeyError):
            raise FileNotFoundError(f"LIDLite5 model not found at: {model_path}")

    return detector.predict(text)


class _HFSequenceClassifierLID:
    """
    Shared loading/inference logic for transformer-based LID classifiers.
    Not meant to be used directly — see LIDNeural5, LIDNeural5_1, LIDNeural25.
    """

    def __init__(self, model_name: str, default_classes=None):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.classes = default_classes

    def load(self):
        """Download and load the model from Hugging Face (runs once; cached after first call)."""
        if self._loaded:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                f"The 'transformers' and 'torch' libraries are required to load {type(self).__name__}. "
                "Install with: pip install olaverse[deeplearning]"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            id2lbl = self.model.config.id2label
            self.classes = [id2lbl[i] if i in id2lbl else id2lbl[str(i)] for i in range(len(id2lbl))]

        self.model.eval()
        self._loaded = True

    def predict_proba(self, text: str) -> dict:
        """
        Return probability distribution over all supported languages.

        Returns:
            dict: {label: probability, ...}
        """
        if not self._loaded:
            self.load()

        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()

        if not isinstance(probs, list):
            probs = [probs]

        return {self.classes[i]: probs[i] for i in range(len(self.classes))}

    def predict(self, text: str) -> str:
        """Predict the dominant language of the text."""
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)

    def predict_proba_batch(self, texts: list) -> list:
        """
        Predict language probabilities for a list of texts in a single forward pass.

        Args:
            texts: List of text strings.

        Returns:
            List of dicts, one per input text: [{label: probability, ...}, ...]
        """
        if not self._loaded:
            self.load()

        import torch

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).tolist()

        return [{self.classes[i]: row[i] for i in range(len(self.classes))} for row in probs]

    def predict_batch(self, texts: list) -> list:
        """
        Predict the dominant language for a list of texts.

        Uses a single batched forward pass — much faster than calling
        ``predict()`` in a loop for large inputs.

        Args:
            texts: List of text strings.

        Returns:
            List of predicted labels.
        """
        proba_list = self.predict_proba_batch(texts)
        return [max(p, key=p.get) for p in proba_list]


class LIDNeural5(_HFSequenceClassifierLID):
    """
    High-accuracy transformer-based language identifier for 5 Nigerian languages.

    Base Model: castorini/afriberta_large (XLM-RoBERTa, 125M parameters)
    Fine-tuned on: Yoruba ('yor'), Hausa ('hau'), Igbo ('ibo'), Pidgin ('pcm'), English ('eng')
    Validation accuracy: 98.96% macro-F1

    Requires: pip install olaverse[deeplearning]
    """

    def __init__(self, model_name="olaverse/lid-neural-5"):
        super().__init__(model_name, default_classes=['eng', 'hau', 'ibo', 'pcm', 'yor'])


class LIDNeural5_1(_HFSequenceClassifierLID):
    """
    Compact language identifier for the 4 main Nigerian languages, built as a
    classification head on olaverse/mist-encoder-base-ng (ModernBERT, ~31M parameters).

    Labels: 'Hausa', 'Yoruba', 'Igbo', 'Nigerian Pidgin'.

    No English/'other' class — out-of-set languages (e.g. English) will be
    confidently mislabelled, most often as Nigerian Pidgin. Use LIDLite25 or
    LIDNeural25 instead if inputs may include English or other non-Nigerian
    languages.

    Requires: pip install olaverse[deeplearning]
    """

    def __init__(self, model_name="olaverse/lid-neural-5.1"):
        super().__init__(model_name)


class LIDNeural25(_HFSequenceClassifierLID):
    """
    Transformer-based (XLM-RoBERTa, 125M parameters) language identifier for
    25 languages — higher accuracy than LIDLite25, especially on short text,
    at the cost of needing transformers/torch.

    Two checkpoints for two input lengths (variant=):
        "passages"  — lid-neural-25.1, long-form text (documents, articles)
        "questions" — lid-neural-25.2, short text (queries, chat messages) [default]

    Requires: pip install olaverse[deeplearning]
    """

    _MODEL_IDS = {
        "passages": "olaverse/lid-neural-25.1",
        "questions": "olaverse/lid-neural-25.2",
    }

    def __init__(self, variant: str = "questions"):
        if variant not in self._MODEL_IDS:
            raise ValueError(f"variant must be one of {list(self._MODEL_IDS)}, got {variant!r}")
        self.variant = variant
        super().__init__(self._MODEL_IDS[variant])


class LIDLite25:
    """
    Lightweight, CPU-only fastText language identifier for 25 languages.
    Sub-millisecond inference, ~5-10MB per checkpoint, no GPU required.

    Two checkpoints for two input lengths (variant=):
        "passages"  — long-form text (documents, articles)
        "questions" — short text (queries, chat messages) [default]

    For higher accuracy at the cost of needing transformers/torch, see LIDNeural25.

    Requires: pip install olaverse[lid]
    """

    _CHECKPOINTS = {"passages": "passages.bin", "questions": "questions.bin"}

    def __init__(self, variant: str = "questions"):
        if variant not in self._CHECKPOINTS:
            raise ValueError(f"variant must be one of {list(self._CHECKPOINTS)}, got {variant!r}")
        self.variant = variant
        self._model = None

    def load(self):
        """Download and load the fastText checkpoint (runs once; cached after first call)."""
        if self._model is not None:
            return

        try:
            import fasttext
        except ImportError:
            raise ImportError(
                "The 'fasttext' library is required to load LIDLite25. "
                "Install with: pip install olaverse[lid]"
            )

        model_path = get_model_path(self._CHECKPOINTS[self.variant], repo_id="olaverse/lid-lite-25")
        self._model = fasttext.load_model(model_path)

    def predict_proba(self, text: str) -> dict:
        """
        Return probability distribution over all 25 languages.

        Returns:
            dict: {'eng': 0.99, 'fra': 0.005, ...} (ISO 639-3 codes)
        """
        if self._model is None:
            self.load()

        labels, probs = self._model.predict(text.replace("\n", " ").strip(), k=-1)
        return {label.replace("__label__", ""): float(prob) for label, prob in zip(labels, probs)}

    def predict(self, text: str) -> str:
        """Predict the dominant language of the text (ISO 639-3 code, e.g. 'eng')."""
        probs = self.predict_proba(text)
        return max(probs, key=probs.get)
