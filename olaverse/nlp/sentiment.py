import os
import json
import math
import re
import numpy as np
from olaverse.utils.downloader import get_model_path

_MODEL_CACHE = {}

def _load_model(model_path=None):
    global _MODEL_CACHE
    
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("sentiment_model.json")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "sentiment_model.json")
            
    if resolved_path in _MODEL_CACHE:
        return _MODEL_CACHE[resolved_path]

    if not os.path.exists(resolved_path):
        if model_path is not None:
            raise FileNotFoundError(f"Sentiment model not found at: {model_path}")
        # Fallback values
        return {
            "vocab": {},
            "idf": [],
            "coef": [],
            "intercept": 0.0
        }

    with open(resolved_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)
    _MODEL_CACHE[resolved_path] = model_data
    return model_data

def analyze_sentiment(text, model_path=None):
    """
    Analyze the sentiment of the given text.
    Works across Yoruba, Hausa, Igbo, Pidgin, and English.
    Returns: A dictionary with 'label' ('positive' or 'negative') and 'confidence' (float).
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {"label": "positive", "confidence": 0.5}

    model = _load_model(model_path)
    vocab = model["vocab"]
    idf = np.array(model["idf"])
    coef = np.array(model["coef"])
    intercept = model["intercept"]

    if not vocab or len(idf) == 0:
        return {"label": "positive", "confidence": 0.5}

    # 1. Tokenize text using the same pattern as training: r'\b\w+\b'
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # 2. Vectorize (TF)
    vector = np.zeros(len(vocab))
    for token in tokens:
        if token in vocab:
            vector[vocab[token]] += 1

    # 3. Apply IDF
    vector = vector * idf

    # 4. L2 Normalize
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    # 5. Logistic Regression Inference
    z = np.dot(vector, coef) + intercept
    prob = 1.0 / (1.0 + np.exp(-z))

    if prob >= 0.5:
        return {"label": "positive", "confidence": float(prob)}
    else:
        return {"label": "negative", "confidence": float(1.0 - prob)}
