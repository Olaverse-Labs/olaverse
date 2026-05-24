import os
import json
import math
import re
from olaverse.utils.downloader import get_model_path

_MODEL_CACHE = {}

def _load_model(model_path=None):
    global _MODEL_CACHE
    
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("language_detector.json")
        except Exception:
            # Fallback to local package directory if offline and download fails
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "language_detector.json")
        
    if resolved_path in _MODEL_CACHE:
        return _MODEL_CACHE[resolved_path]

    if not os.path.exists(resolved_path):
        if model_path is not None:
            raise FileNotFoundError(f"Language detector model not found at: {model_path}")
        # Return a fallback model if not trained/found
        return {
            "priors": {"yor": -1.6, "hau": -1.6, "ibo": -1.6, "pcm": -1.6, "eng": -1.6},
            "vocab": {},
            "features": {},
            "default_log_prob": -12.0
        }

    with open(resolved_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)
    _MODEL_CACHE[resolved_path] = model_data
    return model_data

def extract_ngrams(text, n_min=1, n_max=4):
    """
    Extract character n-grams from 1 to 4 characters.
    """
    # Normalize text: lowercase, strip extra spaces, keep alphanumeric and diacritics
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    
    ngrams = []
    # Add start and end boundaries
    padded = f"_{text}_"
    for n in range(n_min, n_max + 1):
        for i in range(len(padded) - n + 1):
            ngrams.append(padded[i:i+n])
    return ngrams

def detect_language(text, model_path=None):
    """
    Detect the language of the given text.
    Returns: 'yor' (Yoruba), 'hau' (Hausa), 'ibo' (Igbo), 'pcm' (Pidgin), or 'eng' (English).
    """
    if not text or not isinstance(text, str) or not text.strip():
        return "eng"  # default fallback

    model = _load_model(model_path)
    priors = model["priors"]
    features = model["features"]
    default_log_prob = model.get("default_log_prob", -12.0)

    # Extract n-grams from input text
    ngrams = extract_ngrams(text)
    
    # Initialize scores with priors
    scores = {lang: log_prior for lang, log_prior in priors.items()}
    
    for ngram in ngrams:
        if ngram in features:
            for lang in scores:
                scores[lang] += features[ngram].get(lang, default_log_prob)
        else:
            for lang in scores:
                scores[lang] += default_log_prob

    # Return the language with the highest score
    return max(scores, key=scores.get)
