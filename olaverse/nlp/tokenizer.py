import os
from tokenizers import Tokenizer as HFTokenizer

class Tokenizer:
    """
    A unified BPE Tokenizer for Nigerian languages.
    Supports 'yoruba', 'igbo', 'hausa', 'pidgin', and 'naija' (unified).
    """
    def __init__(self, lang="naija", model_path=None):
        self.lang = lang.strip()
        lang_lower = self.lang.lower()
        
        # Map lang names to JSON files
        mapping = {
            "yoruba": "otk-bpe-50k-yo.json",
            "yo": "otk-bpe-50k-yo.json",
            "igbo": "otk-bpe-50k-ig.json",
            "ig": "otk-bpe-50k-ig.json",
            "hausa": "otk-bpe-50k-ha.json",
            "ha": "otk-bpe-50k-ha.json",
            "pidgin": "otk-bpe-50k-pcm.json",
            "pcm": "otk-bpe-50k-pcm.json",
            "naija": "otk-bpe-50k-naija.json"
        }
        
        if lang_lower in mapping:
            model_filename = mapping[lang_lower]
        else:
            # Allow loading directly by model name (e.g., "otk-bpe-50k-yo")
            model_filename = self.lang
            if not model_filename.endswith(".json"):
                model_filename += ".json"
        
        resolved_path = model_path
        if resolved_path is not None:
            if not os.path.exists(resolved_path) and not resolved_path.endswith(".json") and os.path.exists(resolved_path + ".json"):
                resolved_path += ".json"
                
        if resolved_path is None:
            from olaverse.utils.downloader import get_model_path
            try:
                resolved_path = get_model_path(model_filename)
            except Exception:
                resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", model_filename)
        
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Tokenizer model file not found: {resolved_path}")
            
        self._tokenizer = HFTokenizer.from_file(resolved_path)

    def encode(self, text):
        """
        Encode input text into a list of token IDs.
        """
        if not text or not isinstance(text, str):
            return []
        output = self._tokenizer.encode(text)
        return output.ids

    def decode(self, ids):
        """
        Decode a list of token IDs back into a string.
        """
        if not ids or not isinstance(ids, list):
            return ""
        return self._tokenizer.decode(ids)
