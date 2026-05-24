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
            resolved_path = get_model_path(model_name_or_path)
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
