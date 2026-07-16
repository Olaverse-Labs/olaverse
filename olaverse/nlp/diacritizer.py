"""
DiacNet — Runtime Diacritization Engine
=======================================
Provides a unified Diacritizer class to load and use any of the available
DiacNet models for African languages.

Supported Methods for Yoruba ('yo'):
  - "viterbi": Fast statistical n-gram Viterbi decoder (default)
  - "knn": Character k-NN backoff (dot-below only)
  - "bilstm": High-accuracy character-level BiLSTM
  - "transformer": High-accuracy XLM-RoBERTa Transformer

Supported Methods for Igbo ('ig'):
  - "knn": Character k-NN backoff (default)
"""

import os
import json
import unicodedata
import re
from olaverse.utils.downloader import get_model_path

_YORUBA_MODEL_CACHE = {}
_YORUBA_DB_MODEL_CACHE = {}
_IGBO_MODEL_CACHE = {}
_NEURAL_CACHE = {}

def remove_tones(text):
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(
        c for c in decomposed 
        if unicodedata.category(c) != 'Mn' or ord(c) == 0x0323
    )
    return unicodedata.normalize('NFC', filtered)

def strip_all_diacritics(text):
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(
        c for c in decomposed 
        if unicodedata.category(c) != 'Mn'
    )
    return unicodedata.normalize('NFC', filtered)

def _load_diacritizer_model(path, is_custom=False):
    if not os.path.exists(path):
        if is_custom:
            raise FileNotFoundError(f"Diacritizer model file not found at: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── Viterbi Decoder ───
def viterbi_decode(text, model):
    candidates_map = model.get("candidates", {})
    transitions = model.get("transitions", {})
    unigrams = model.get("unigrams", {})
    
    tokens = re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip() and re.match(r'^[\w\u0300-\u036f]+$', t)]
    
    if not word_indices:
        return text
        
    dp = []
    first_word_idx = word_indices[0]
    first_token = tokens[first_word_idx]
    first_token_lower = first_token.lower()
    
    candidates = candidates_map.get(first_token_lower, [first_token_lower])
    first_dp = {}
    for cand in candidates:
        if first_token.isupper():
            cand_formatted = cand.upper()
        elif first_token[0].isupper():
            cand_formatted = cand.capitalize()
        else:
            cand_formatted = cand
            
        unigram_prob = unigrams.get(cand, -12.0)
        first_dp[cand_formatted] = (unigram_prob, None)
    dp.append(first_dp)
    
    for step_idx in range(1, len(word_indices)):
        prev_word_idx = word_indices[step_idx - 1]
        curr_word_idx = word_indices[step_idx]
        curr_token = tokens[curr_word_idx]
        curr_token_lower = curr_token.lower()
        
        curr_candidates = candidates_map.get(curr_token_lower, [curr_token_lower])
        curr_dp = {}
        for cand in curr_candidates:
            if curr_token.isupper():
                cand_formatted = cand.upper()
            elif curr_token[0].isupper():
                cand_formatted = cand.capitalize()
            else:
                cand_formatted = cand
                
            best_prob = -float('inf')
            best_prev = None
            
            for prev_cand, (prev_prob, _) in dp[-1].items():
                prev_cand_lower = prev_cand.lower()
                transition_key = f"{prev_cand_lower} {cand}"
                trans_prob = transitions.get(transition_key, unigrams.get(cand, -12.0) - 5.0)
                
                total_prob = prev_prob + trans_prob
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_prev = prev_cand
                    
            curr_dp[cand_formatted] = (best_prob, best_prev)
        dp.append(curr_dp)
        
    best_final_prob = -float('inf')
    best_final_cand = None
    for cand, (prob, _) in dp[-1].items():
        if prob > best_final_prob:
            best_final_prob = prob
            best_final_cand = cand
            
    best_path = [best_final_cand]
    for step_idx in range(len(word_indices) - 1, 0, -1):
        prev_cand = dp[step_idx][best_path[-1]][1]
        best_path.append(prev_cand)
        
    best_path.reverse()
    
    output_tokens = list(tokens)
    for idx, word_idx in enumerate(word_indices):
        output_tokens[word_idx] = best_path[idx]
        
    return "".join(output_tokens)

# ─── KNN Decoder ───
def knn_decode(text, model, target_chars):
    db_5 = model.get("db_5", {})
    db_3 = model.get("db_3", {})
    db_1 = model.get("db_1", {})
    
    tokens = re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip() and re.match(r'^[\w\u0300-\u036f]+$', t)]
    
    if not word_indices:
        return text
        
    def get_context(word, i, W=2):
        left = word[max(0, i-W) : i]
        left = "_" * (W - len(left)) + left
        right = word[i+1 : min(len(word), i+1+W)]
        right = right + "_" * (W - len(right))
        return left + word[i] + right
        
    def predict_backoff(ctx_query):
        if ctx_query in db_5: return db_5[ctx_query]
        ctx_3 = ctx_query[1:4]
        if ctx_3 in db_3: return db_3[ctx_3]
        target = ctx_query[2]
        return db_1.get(target, target)
        
    output_tokens = list(tokens)
    for word_idx in word_indices:
        token = tokens[word_idx]
        token_lower = token.lower()
        
        pred_chars = list(token_lower)
        for i in range(len(token_lower)):
            if token_lower[i] in target_chars:
                ctx_query = get_context(token_lower, i, 2)
                pred_chars[i] = predict_backoff(ctx_query)
                
        pred_word = "".join(pred_chars)
        if token.isupper():
            pred_word_formatted = pred_word.upper()
        elif token[0].isupper():
            pred_word_formatted = pred_word.capitalize()
        else:
            pred_word_formatted = pred_word
            
        output_tokens[word_idx] = pred_word_formatted
        
    return "".join(output_tokens)

# ─── Neural Decoders ───
class BiLSTMDecoder:
    def __init__(self, pt_path, vocab_path):
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        
        class DiacNetCharModel(nn.Module):
            def __init__(self, vocab_size, emb_dim=64, hidden_dim=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True, num_layers=2)
                self.classifier = nn.Linear(hidden_dim, 6)
            def forward(self, char_seqs, lengths):
                emb = self.embedding(char_seqs)
                packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
                out, _ = self.lstm(packed)
                out, _ = pad_packed_sequence(out, batch_first=True)
                return self.classifier(out)

        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
        self.char_vocab = v_data["char_vocab"]
        self.word_candidates = v_data["word_candidates"]
        
        state = torch.load(pt_path, map_location=self.device)
        self.model = DiacNetCharModel(len(self.char_vocab)).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

    def decode(self, text):
        import torch
        text_nfd = unicodedata.normalize('NFD', text)
        base_chars = [c for c in text_nfd if unicodedata.category(c) != 'Mn']
        if not base_chars: return text
        
        char_ids = [self.char_vocab.get(c, 1) for c in base_chars]
        tensor_in = torch.tensor([char_ids], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(char_ids)], dtype=torch.long)
        
        with torch.no_grad():
            logits = self.model(tensor_in, lengths)
            preds = logits.argmax(dim=-1).squeeze(0).tolist()
            
        parts = []
        for c, l in zip(base_chars, preds):
            if l == 0 or not c.isalpha(): parts.append(c)
            elif l == 1: parts.append(c + '\u0323')
            elif l == 2: parts.append(c + '\u0301')
            elif l == 3: parts.append(c + '\u0300')
            elif l == 4: parts.append(c + '\u0323\u0301')
            elif l == 5: parts.append(c + '\u0323\u0300')
        pred_sentence = unicodedata.normalize('NFC', "".join(parts))
        
        plain_words = re.findall(r'\S+', strip_all_diacritics(text))
        pred_words = re.findall(r'\S+', pred_sentence)
        tokens = re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)
        word_indices = [i for i, t in enumerate(tokens) if t.strip() and re.match(r'^[\w\u0300-\u036f]+$', t)]
        
        for w_idx, pw, pw_pred in zip(word_indices, plain_words, pred_words):
            pw_l = pw.lower()
            cands = self.word_candidates.get(pw_l, [])
            if not cands or pw_pred.lower() in cands:
                corrected = pw_pred
            else:
                corrected = cands[0]
                if pw_pred and pw_pred[0].isupper():
                    corrected = corrected.capitalize()
            tokens[w_idx] = corrected
            
        return "".join(tokens)

_DIACNET_LANG_TAGS = {
    "yo": "<yor>", "vi": "<vie>", "ig": "<ibo>", "ha": "<hau>", "pl": "<pol>",
    "tr": "<tur>", "pt": "<por>", "es": "<spa>", "fr": "<fra>", "it": "<ita>",
}

class DiacNetDecoder:
    """diacnet-1.0 diacritic restoration (byte-level seq2seq) — one joint model, 10 languages."""

    def __init__(self, model_name="olaverse/diacnet-1.0"):
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()

    def decode(self, text: str, lang: str = "yo", max_new_tokens: int = 256) -> str:
        import torch

        tag = _DIACNET_LANG_TAGS.get(lang.lower())
        if tag is None:
            raise ValueError(
                f"Unsupported language '{lang}' for diacnet-1.0. "
                f"Supported: {sorted(_DIACNET_LANG_TAGS)}"
            )

        inputs = self.tokenizer(f"{tag} {text}", return_tensors="pt")
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

class TransformerDecoder:
    def __init__(self, pt_path, vocab_path):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
        self.word_candidates = v_data["word_candidates"]
        base_model = v_data["base_model"]
        
        class DiacNetYorXModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(base_model)
                self.classifier = nn.Linear(self.encoder.config.hidden_size, 8)
            def forward(self, input_ids, attention_mask):
                return self.classifier(self.encoder(input_ids, attention_mask).last_hidden_state)
                
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = DiacNetYorXModel().to(self.device)
        
        state = torch.load(pt_path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

    def decode(self, text):
        import torch
        plain_words = re.findall(r'\S+', strip_all_diacritics(text))
        if not plain_words: return text
        
        encoding = self.tokenizer(plain_words, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=128)
        input_ids = encoding['input_ids'].to(self.device)
        attn_mask = encoding['attention_mask'].to(self.device)
        word_ids = encoding.word_ids()
        
        with torch.no_grad():
            logits = self.model(input_ids, attn_mask)
            preds = logits.argmax(dim=-1).squeeze(0)
            
        corrected_words = []
        prev_wid = None
        for j, wid in enumerate(word_ids):
            if wid is None or wid == prev_wid: continue
            prev_wid = wid
            pw = plain_words[wid].lower()
            cands = self.word_candidates.get(pw, [pw])
            pred_idx = preds[j].item()
            if pred_idx < len(cands):
                pred_label = cands[pred_idx]
            else:
                pred_label = cands[0]
                
            if plain_words[wid].isupper():
                pred_label = pred_label.upper()
            elif plain_words[wid][0].isupper():
                pred_label = pred_label.capitalize()
            corrected_words.append(pred_label)
            
        tokens = re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)
        word_indices = [i for i, t in enumerate(tokens) if t.strip() and re.match(r'^[\w\u0300-\u036f]+$', t)]
        for w_idx, corrected in zip(word_indices, corrected_words):
            tokens[w_idx] = corrected
            
        return "".join(tokens)

# ─── Legacy Functions ───
def diacritize_yoruba(text, model_path=None):
    global _YORUBA_MODEL_CACHE
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("yoruba_diacritizer.json", repo_id="olaverse/diacnet-yor-viterbi")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yoruba_diacritizer.json")
    if resolved_path in _YORUBA_MODEL_CACHE:
        model = _YORUBA_MODEL_CACHE[resolved_path]
    else:
        model = _load_diacritizer_model(resolved_path, is_custom=(model_path is not None))
        _YORUBA_MODEL_CACHE[resolved_path] = model
    return viterbi_decode(text, model)

def diacritize_yoruba_dot_below(text, model_path=None):
    global _YORUBA_DB_MODEL_CACHE
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("yoruba_diacritizer_dot_below.json", repo_id="olaverse/diacnet-yor-db")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yoruba_diacritizer_dot_below.json")
    if resolved_path in _YORUBA_DB_MODEL_CACHE:
        model = _YORUBA_DB_MODEL_CACHE[resolved_path]
    else:
        model = _load_diacritizer_model(resolved_path, is_custom=(model_path is not None))
        _YORUBA_DB_MODEL_CACHE[resolved_path] = model
    return knn_decode(text, model, {'o', 'e', 's'})

def diacritize_igbo(text, model_path=None):
    global _IGBO_MODEL_CACHE
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("igbo_diacritizer.json", repo_id="olaverse/diacnet-ig")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "igbo_diacritizer.json")
    if resolved_path in _IGBO_MODEL_CACHE:
        model = _IGBO_MODEL_CACHE[resolved_path]
    else:
        model = _load_diacritizer_model(resolved_path, is_custom=(model_path is not None))
        _IGBO_MODEL_CACHE[resolved_path] = model
    return knn_decode(text, model, {'i', 'u', 'o', 'e'})

# ─── Unified Wrapper Class ───
MODEL_REGISTRY = {
    "diacnet-yor-viterbi": {"lang": "yo", "method": "viterbi"},
    "diacnet-yor-db":      {"lang": "yo", "method": "knn"},
    "diacnet-ig":          {"lang": "ig", "method": "knn"},
    "diacnet-yor":         {"lang": "yo", "method": "bilstm"},
    "diacnet-yor-x":       {"lang": "yo", "method": "transformer"},
    "diacnet-1.0":         {"lang": "multi", "method": "diacnet"},
    "auto":                {"lang": "auto", "method": "auto"},
}

class Diacritizer:
    """
    Unified interface for restoring diacritics in African languages.

    Pass a model ID to use a specific backend, or ``model="auto"`` to detect
    the language automatically and route to the appropriate diacritizer.

    Args:
        model: One of:

            * ``"diacnet-yor-viterbi"`` — Yoruba, fast Viterbi n-gram (default)
            * ``"diacnet-yor-db"``      — Yoruba dot-below only, KNN
            * ``"diacnet-ig"``          — Igbo, KNN
            * ``"diacnet-yor"``         — Yoruba BiLSTM (requires ``olaverse[deeplearning]``)
            * ``"diacnet-yor-x"``       — Yoruba XLM-RoBERTa (requires ``olaverse[deeplearning]``)
            * ``"diacnet-1.0"``         — Multilingual DiacNet, 10 languages, see ``lang=``
                                          (requires ``olaverse[deeplearning]``)
            * ``"auto"``                — detect language via LIDLite5, then route automatically

        lang: Target language for ``"diacnet-1.0"`` only. One of
              ``"yo", "vi", "ig", "ha", "pl", "tr", "pt", "es", "fr", "it"``. Ignored
              by every other model.
    """

    def __init__(self, model: str = "diacnet-yor-viterbi", lang: str = None):
        if model not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model}' is not recognised. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        config = MODEL_REGISTRY[model]
        self.lang = config["lang"]
        self.method = config["method"]
        self.neural_decoder = None
        self.diacnet_lang = lang or "yo"

        # Auto-routing: lazy-load LIDLite5 + sub-diacritizers at restore() time
        if self.method == "auto":
            self._lid = None
            self._sub: dict = {}
            return

        if self.method == "bilstm":
            if "bilstm" not in _NEURAL_CACHE:
                pt    = get_model_path("diacnet_yor.pt",       repo_id="olaverse/diacnet-yor")
                vocab = get_model_path("diacnet_yor_vocab.json", repo_id="olaverse/diacnet-yor")
                _NEURAL_CACHE["bilstm"] = BiLSTMDecoder(pt, vocab)
            self.neural_decoder = _NEURAL_CACHE["bilstm"]

        elif self.method == "transformer":
            if "transformer" not in _NEURAL_CACHE:
                pt    = get_model_path("diacnet_yor_x.pt",       repo_id="olaverse/diacnet-yor-x")
                vocab = get_model_path("diacnet_yor_x_vocab.json", repo_id="olaverse/diacnet-yor-x")
                _NEURAL_CACHE["transformer"] = TransformerDecoder(pt, vocab)
            self.neural_decoder = _NEURAL_CACHE["transformer"]

        elif self.method == "diacnet":
            # Cache per version key ("diacnet-1.0", later "diacnet-1.1", ...) so
            # future DiacNet releases are one registry line, same decoder class.
            if model not in _NEURAL_CACHE:
                _NEURAL_CACHE[model] = DiacNetDecoder(f"olaverse/{model}")
            self.neural_decoder = _NEURAL_CACHE[model]

    def _auto_restore(self, text: str) -> str:
        """Detect language then delegate to the correct diacritizer."""
        if self._lid is None:
            from olaverse.nlp.language_detection import LIDLite5
            self._lid = LIDLite5()

        lang = self._lid.predict(text)

        if lang == "ibo":
            if "ig" not in self._sub:
                self._sub["ig"] = Diacritizer(model="diacnet-ig")
            return self._sub["ig"].restore(text)

        # Default to Yoruba for 'yor' and any other detected language
        if "yo" not in self._sub:
            self._sub["yo"] = Diacritizer(model="diacnet-yor-viterbi")
        return self._sub["yo"].restore(text)

    def restore(self, text: str) -> str:
        """
        Restore diacritics in the given text.

        Args:
            text: Plain text (tones/diacritics stripped or missing).

        Returns:
            Text with diacritics restored.
        """
        if self.method == "auto":
            return self._auto_restore(text)

        if self.method == "diacnet":
            return self.neural_decoder.decode(text, lang=self.diacnet_lang)

        if self.neural_decoder:
            return self.neural_decoder.decode(text)

        if self.lang == "yo":
            if self.method == "viterbi":
                return diacritize_yoruba(text)
            elif self.method == "knn":
                return diacritize_yoruba_dot_below(text)
            else:
                raise ValueError(f"Unsupported method '{self.method}' for Yoruba.")

        if self.lang == "ig":
            return diacritize_igbo(text)

        raise ValueError(f"Unsupported language '{self.lang}'.")
