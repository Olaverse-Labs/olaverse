import os
from tokenizers import Tokenizer as HFTokenizer

_NIGERIAN_REPO = "olaverse/otk-bpe-50k"
_MULTILINGUAL_REPO = "olaverse/otk-bpe"

# Nigerian-languages family (fixed 50k vocab) — root-level combined JSON files
_NIGERIAN_MAPPING = {
    "yoruba": "otk-bpe-50k-yo.json",
    "yo": "otk-bpe-50k-yo.json",
    "igbo": "otk-bpe-50k-ig.json",
    "ig": "otk-bpe-50k-ig.json",
    "hausa": "otk-bpe-50k-ha.json",
    "ha": "otk-bpe-50k-ha.json",
    "pidgin": "otk-bpe-50k-pcm.json",
    "pcm": "otk-bpe-50k-pcm.json",
    "naija": "otk-bpe-50k-naija.json",
}

# Multilingual family (Swahili / Kinyarwanda / merged) — subfolder/tokenizer.json layout
_MULTILINGUAL_VARIANTS = {
    "sw-50k", "sw-100k", "sw-150k",
    "kin-50k", "kin-100k", "kin-150k",
    "merged-50k", "merged-100k", "merged-150k",
}

class Tokenizer:
    """
    A unified BPE Tokenizer for African languages.

    Nigerian family (fixed 50k vocab): 'yoruba'/'yo', 'igbo'/'ig', 'hausa'/'ha',
    'pidgin'/'pcm', and 'naija' (unified).

    Multilingual family (50k/100k/150k vocab, see olaverse/otk-bpe): 'sw-50k',
    'sw-100k', 'sw-150k' (Swahili); 'kin-50k', 'kin-100k', 'kin-150k' (Kinyarwanda);
    'merged-50k', 'merged-100k', 'merged-150k' (French + Kinyarwanda + English + Swahili).
    """
    def __init__(self, lang="naija", model_path=None):
        self.lang = lang.strip()
        lang_lower = self.lang.lower()

        if lang_lower in _NIGERIAN_MAPPING:
            model_filename = _NIGERIAN_MAPPING[lang_lower]
            repo_id = _NIGERIAN_REPO
        elif lang_lower in _MULTILINGUAL_VARIANTS:
            model_filename = f"{lang_lower}/tokenizer.json"
            repo_id = _MULTILINGUAL_REPO
        else:
            # Allow loading directly by model name (e.g., "otk-bpe-50k-yo")
            model_filename = self.lang
            if not model_filename.endswith(".json"):
                model_filename += ".json"
            repo_id = _NIGERIAN_REPO

        resolved_path = model_path
        if resolved_path is not None:
            if not os.path.exists(resolved_path) and not resolved_path.endswith(".json") and os.path.exists(resolved_path + ".json"):
                resolved_path += ".json"

        if resolved_path is None:
            from olaverse.utils.downloader import get_model_path
            try:
                resolved_path = get_model_path(model_filename, repo_id=repo_id)
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
