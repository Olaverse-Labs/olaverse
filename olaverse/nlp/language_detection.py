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
    Not meant to be used directly — see LIDNeural5, LIDNeural5_1, LIDNeural25,
    LIDNeural608.

    Subclasses can change behaviour through four small hooks: ``max_length``
    (tokenizer truncation), ``_prepare`` (text clean-up before tokenizing),
    ``_from_pretrained_kwargs`` (extra loading options) and ``_probs`` (logits
    to probabilities). The defaults reproduce the original behaviour exactly.
    """

    #: Tokenizer truncation length.
    max_length = 128

    def __init__(self, model_name: str, default_classes=None, device: str = None):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.classes = default_classes
        self._device = device      # None leaves the model where transformers put it

    # -- hooks ------------------------------------------------------------
    def _prepare(self, text: str) -> str:
        return text

    def _from_pretrained_kwargs(self) -> dict:
        return {}

    def _after_load(self) -> None:
        """Called once the model and ``classes`` are set, before ``eval()``."""

    def _probs(self, logits):
        import torch
        return torch.softmax(logits, dim=-1)

    def _encode(self, texts):
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            **({"padding": True} if isinstance(texts, list) else {}),
        )
        return enc.to(self._device) if self._device else enc

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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **self._from_pretrained_kwargs())
        if self._device:
            self.model.to(self._device)

        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            id2lbl = self.model.config.id2label
            self.classes = [id2lbl[i] if i in id2lbl else id2lbl[str(i)] for i in range(len(id2lbl))]

        self._after_load()
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

        inputs = self._encode(self._prepare(text))

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self._probs(logits).squeeze().tolist()

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

        inputs = self._encode([self._prepare(t) for t in texts])

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self._probs(logits).tolist()

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


# ---------------------------------------------------------------------------
# 608-language identification: lid-lite-608 (fastText) / lid-neural-608 (mmBERT)
# ---------------------------------------------------------------------------
# The two models share their 608 languages, their labels ("yor_Latn": ISO 639-3
# plus ISO 15924 script, and a noise class "zxx_Zxxx") and their two reading
# modes, so the mode logic lives here once.
#
#   coverage  every language equally likely — corpus building, low-resource mining
#   traffic   scores shifted by alpha * log_prior_ratio (priors.json, identical in
#             both repos) so short or ambiguous input leans towards the languages
#             that dominate real traffic — user input, routing

_LID608_MODES = ("coverage", "traffic")
_LID608_PRIORS_REPO = "olaverse/lid-lite-608"
_LID608_NOISE_LABEL = "zxx_Zxxx"


def _check_608_mode(mode: str) -> str:
    if mode not in _LID608_MODES:
        raise ValueError(f"mode must be one of {list(_LID608_MODES)}, got {mode!r}")
    return mode


def _clean_608(text) -> str:
    """Collapse whitespace to a single line, as both models were trained and scored on.

    fastText reads one line at a time, so a newline would silently truncate the input.
    """
    cleaned = " ".join(str(text if text is not None else "").split())
    if not cleaned:
        raise ValueError("text is empty — there is nothing to identify the language of.")
    return cleaned


def _load_priors_608(repo_id: str) -> dict:
    """``{label: alpha * log_prior_ratio}`` for traffic mode."""
    path = get_model_path("priors.json", repo_id=repo_id)
    with open(path, "r", encoding="utf-8") as f:
        priors = json.load(f)
    alpha = float(priors["alpha"])
    return {label: alpha * float(v) for label, v in priors["log_prior_ratio"].items()}


class LIDLite608:
    """
    Fast, CPU-only fastText language identifier for **608 languages**, African-first.

    37 MB, ~6,800 texts/s on one CPU thread. Labels are ISO 639-3 plus ISO 15924
    script — ``'yor_Latn'``, ``'srp_Cyrl'`` — and ``'zxx_Zxxx'`` marks numbers, URLs,
    code and other non-language text. (The 5- and 25-language classes return bare
    codes like ``'yor'``; use ``label.split('_')[0]`` for the code alone.)

    Two modes in one model (``mode=``):
        "coverage" — every language equally likely; best per-language accuracy.
                     Use it to build corpora or mine low-resource text. [default]
        "traffic"  — scores shifted by each language's real-world frequency, so
                     short or ambiguous input leans towards common languages. Use
                     it for user input, chat and routing. In this mode the
                     probabilities are spread over the 40 likeliest candidates.

    For higher accuracy on short and conversational text see :class:`LIDNeural608`.

    Requires: pip install olaverse[lid]

    Quick start:
        >>> lid = LIDLite608()
        >>> lid.predict("Ẹ kú àárọ̀, ṣé dáadáa ni?")
        'yor_Latn'
        >>> LIDLite608(mode="traffic").predict("Bonjour mon ami")
        'fra_Latn'
    """

    REPO_ID = "olaverse/lid-lite-608"
    #: How many fastText candidates traffic mode re-scores.
    TRAFFIC_CANDIDATES = 40
    _LABEL_PREFIX = "__label__"

    def __init__(self, mode: str = "coverage"):
        self.mode = _check_608_mode(mode)
        self._model = None
        self._bias = None

    def load(self):
        """Download and load the fastText model and priors (runs once; cached afterwards)."""
        if self._model is not None:
            return

        try:
            import fasttext
        except ImportError:
            raise ImportError(
                "The 'fasttext' library is required to load LIDLite608. "
                "Install with: pip install olaverse[lid]"
            )

        model_path = get_model_path("model.ftz", repo_id=self.REPO_ID)
        self._bias = _load_priors_608(self.REPO_ID) if self.mode == "traffic" else {}
        self._model = fasttext.load_model(model_path)

    def _candidates(self, text: str, k: int):
        """The model's ``k`` likeliest ``(label, probability)`` pairs (``k=-1``: all)."""
        text = _clean_608(text)
        try:
            labels, probs = self._model.predict(text, k=k)
            pairs = list(zip(labels, (float(p) for p in probs)))
        except ValueError:
            # fasttext-wheel's predict() calls np.array(..., copy=False), which
            # NumPy >= 2 rejects. The binding underneath has no NumPy in it.
            pairs = [(label, float(p))
                     for p, label in self._model.f.predict(text + "\n", k, 0.0, "strict")]
        return [(label[len(self._LABEL_PREFIX):] if label.startswith(self._LABEL_PREFIX) else label, p)
                for label, p in pairs]

    def _scored(self, text: str):
        """``[(label, log score)]`` over the traffic-mode candidates, best first."""
        scored = [(label, math.log(max(p, 1e-12)) + self._bias.get(label, 0.0))
                  for label, p in self._candidates(text, self.TRAFFIC_CANDIDATES)]
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def predict_proba(self, text: str, top_k: int = None) -> dict:
        """
        Probabilities for the likeliest languages, best first.

        Args:
            text: Any length, from one word to a document. Newlines are collapsed.
            top_k: Keep only the ``top_k`` likeliest labels. ``None`` returns all
                   608 in coverage mode and the 40 candidates in traffic mode.

        Returns:
            dict: ``{'yor_Latn': 0.99, 'ibo_Latn': 0.004, ...}``

        Raises:
            ValueError: if ``text`` is empty.
        """
        if self._model is None:
            self.load()

        if self.mode == "coverage":
            pairs = self._candidates(text, -1 if top_k is None else top_k)
            return dict(pairs)

        scored = self._scored(text)
        if not scored:
            return {}
        top = max(score for _, score in scored)
        weights = [(label, math.exp(score - top)) for label, score in scored]
        total = sum(w for _, w in weights)
        probs = {label: w / total for label, w in weights}
        return dict(list(probs.items())[:top_k]) if top_k is not None else probs

    def predict(self, text: str) -> str:
        """The likeliest language of ``text`` (e.g. ``'yor_Latn'``)."""
        if self._model is None:
            self.load()
        # No candidates at all means nothing recognisable as language in the text.
        ranked = (self._candidates(text, 1) if self.mode == "coverage"
                  else self._scored(text))
        return ranked[0][0] if ranked else _LID608_NOISE_LABEL

    def predict_batch(self, texts: list) -> list:
        """The likeliest language for each text. Calls :meth:`predict` in turn."""
        return [self.predict(t) for t in texts]


class LIDNeural608(_HFSequenceClassifierLID):
    """
    Transformer language identifier for **608 languages**, African-first — the
    most accurate of the Olaverse LID models on short and conversational text.

    A ModernBERT sequence classifier fine-tuned from ``jhu-clsp/mmBERT-small``
    (140M parameters). Reads up to 512 tokens (about 200 words); longer text is
    truncated, so split long documents into ~200-word chunks. Labels are ISO 639-3
    plus ISO 15924 script — ``'yor_Latn'``, ``'srp_Cyrl'`` — and ``'zxx_Zxxx'``
    marks numbers, URLs, code and other non-language text. (The 5- and
    25-language classes return bare codes like ``'yor'``; use
    ``label.split('_')[0]`` for the code alone.)

    Two modes in one model (``mode=``):
        "coverage" — every language equally likely; best per-language accuracy.
                     Use it to build corpora or mine low-resource text. [default]
        "traffic"  — probabilities shifted by each language's real-world
                     frequency, so short or ambiguous input leans towards common
                     languages. Use it for user input, chat and routing.

    Runs on CPU, but a GPU helps: on CUDA the weights load in bfloat16.

    Requires: pip install olaverse[deeplearning]. The model card asks for
    ``transformers>=5.14``; an older version can fail while loading.

    For a 37 MB CPU-only alternative with the same languages and modes, see
    :class:`LIDLite608`.

    Quick start:
        >>> lid = LIDNeural608()
        >>> lid.predict("Habari za asubuhi")
        'swh_Latn'
        >>> LIDNeural608(mode="traffic").predict_batch(["Good morning", "Mo fẹ́ lọ sí ọjà"])
        ['eng_Latn', 'yor_Latn']
    """

    max_length = 512
    REPO_ID = "olaverse/lid-neural-608"

    def __init__(self, mode: str = "coverage", device: str = None,
                 model_name: str = REPO_ID):
        """
        Args:
            mode: ``"coverage"`` (default) or ``"traffic"``.
            device: ``"cpu"``, ``"cuda"``, ``"mps"`` ... ``None`` or ``"auto"`` picks
                    CUDA when it is available, else CPU.
            model_name: Hugging Face model id.
        """
        super().__init__(model_name, device=device)
        self.mode = _check_608_mode(mode)
        self._bias = None

    # -- hooks ------------------------------------------------------------
    def _prepare(self, text: str) -> str:
        return _clean_608(text)

    def _resolve_device(self):
        import torch
        if self._device in (None, "auto"):
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _from_pretrained_kwargs(self) -> dict:
        import torch
        device = self._resolve_device()
        # bfloat16 on CUDA only (the model card's recommendation); CPU stays float32.
        return {"dtype": torch.bfloat16} if str(device).startswith("cuda") else {}

    def _after_load(self) -> None:
        import torch
        if self.mode == "traffic":
            priors = _load_priors_608(_LID608_PRIORS_REPO)
            self._bias = torch.tensor(
                [priors.get(label, 0.0) for label in self.classes], dtype=torch.float32
            ).to(self._device)

    def _probs(self, logits):
        import torch
        if self.mode != "traffic":
            return torch.softmax(logits.float(), dim=-1)
        # argmax(log p + bias), renormalised so the values are still probabilities
        return torch.softmax(torch.log_softmax(logits.float(), dim=-1) + self._bias, dim=-1)

    def load(self):
        """Download and load the model (runs once; cached after first call)."""
        if self._loaded:
            return
        try:
            super().load()
        except ImportError:
            raise
        except Exception as exc:
            import transformers
            version = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
            if version < (5, 14):
                raise RuntimeError(
                    f"LIDNeural608 could not be loaded with transformers "
                    f"{transformers.__version__}; its model card requires "
                    f"transformers>=5.14 (pip install -U transformers). "
                    f"Original error: {exc}"
                ) from exc
            raise

    def predict_proba_batch(self, texts: list) -> list:
        """Probabilities for many texts in one forward pass (see the base class)."""
        if not texts:
            return []
        return super().predict_proba_batch(texts)
