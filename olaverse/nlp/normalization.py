"""
TTS Text Normalization
======================
Handles text normalization specifically geared towards Text-to-Speech (TTS).
This includes expanding abbreviations and numbers to their spoken forms.
"""

import re

# Basic abbreviation expansion dictionary
# This can be expanded significantly as the models grow.
ABBREVIATIONS = {
    "yo": {
        "mr.": "Míṣìtà",
        "mrs.": "Mísìsì",
        "dr.": "Dọ́kítà",
        "prof.": "Pùrọ̀fẹ́sọ̀",
        "st.": "Sẹ́nítì",
    },
    "ig": {
        "mr.": "Mista",
        "mrs.": "Misis",
        "dr.": "Dọkịta",
        "prof.": "Purofesọ",
        "st.": "Senti",
    }
}

# Basic single-digit spoken forms
DIGITS = {
    "yo": {
        "0": "òdo", "1": "ọ̀kan", "2": "éjì", "3": "ẹ́ta", "4": "ẹ́rin",
        "5": "árùn-ún", "6": "ẹ́fà", "7": "éje", "8": "ẹ́jọ", "9": "ẹ́sàn-án"
    },
    "ig": {
        "0": "efu", "1": "otu", "2": "abụo", "3": "atọ", "4": "anọ",
        "5": "ise", "6": "isi", "7": "asaa", "8": "asatọ", "9": "itoolu"
    }
}

class TTSNormalizer:
    """
    Normalizes text for TTS processing by expanding numbers, symbols, and abbreviations
    into full phonetic/spoken words based on the target language.
    """
    def __init__(self, lang="yo"):
        self.lang = lang.lower()
        self.abbrev_map = ABBREVIATIONS.get(self.lang, {})
        self.digit_map = DIGITS.get(self.lang, {})
        
        # Build regex for abbreviations if any exist
        if self.abbrev_map:
            # Sort keys by length descending to match longest abbreviations first
            escaped_keys = [re.escape(k) for k in sorted(self.abbrev_map.keys(), key=len, reverse=True)]
            # Match word boundary at the start, but use a negative lookahead at the end
            # because "\b" fails if the abbreviation ends in a punctuation mark like ".".
            pattern = r'\b(?:' + '|'.join(escaped_keys) + r')(?!\w)'
            self.abbrev_re = re.compile(pattern, flags=re.IGNORECASE)
        else:
            self.abbrev_re = None

    def _replace_abbreviations(self, match):
        word = match.group(0).lower()
        return self.abbrev_map.get(word, match.group(0))

    def expand_abbreviations(self, text):
        if not text or not self.abbrev_re:
            return text
        return self.abbrev_re.sub(self._replace_abbreviations, text)

    def expand_numbers(self, text):
        """
        Expands digits into spoken words.
        Currently implements a basic digit-by-digit reading as a fallback.
        Can be extended to handle full localized number expansion logic.
        """
        if not text or not self.digit_map:
            return text
            
        def _replace_digit(match):
            digit = match.group(0)
            return " " + self.digit_map.get(digit, digit) + " "
            
        expanded = re.sub(r'\d', _replace_digit, text)
        return re.sub(r'\s+', ' ', expanded).strip()

    def normalize(self, text):
        """
        Applies all TTS normalization pipelines sequentially.
        """
        if not text:
            return ""
        text = self.expand_abbreviations(text)
        text = self.expand_numbers(text)
        return text
