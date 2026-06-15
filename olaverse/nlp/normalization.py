"""
TTS Text Normalization
======================
Handles text normalization for Text-to-Speech: expanding numbers, abbreviations,
and symbols into their spoken forms for Yoruba, Igbo, and Nigerian Pidgin.
"""

import re

# ── Abbreviation expansion tables ─────────────────────────────────────────────
ABBREVIATIONS = {
    "yo": {
        "mr.":   "Míṣìtà",
        "mrs.":  "Mísìsì",
        "ms.":   "Mísì",
        "dr.":   "Dọ́kítà",
        "prof.": "Pùrọ̀fẹ́sọ̀",
        "st.":   "Sẹ́nítì",
        "lt.":   "Liẹtenanti",
        "gen.":  "Gbogbogbo",
        "govt":  "Ìjọba",
        "govt.": "Ìjọba",
        "no.":   "nọ́mbà",
        "etc.":  "àti bẹ́ẹ̀bẹ́ẹ̀ lọ",
    },
    "ig": {
        "mr.":   "Mista",
        "mrs.":  "Misis",
        "ms.":   "Mis",
        "dr.":   "Dọkịta",
        "prof.": "Purofesọ",
        "st.":   "Senti",
        "lt.":   "Lietinanọ",
        "gen.":  "Jeneral",
        "govt":  "Gọọmentị",
        "govt.": "Gọọmentị",
        "no.":   "nọmba",
        "etc.":  "na ndị ọzọ",
    },
    "pcm": {
        "mr.":   "Mista",
        "mrs.":  "Misis",
        "ms.":   "Mis",
        "dr.":   "Dokita",
        "prof.": "Profesa",
        "st.":   "Senta",
        "lt.":   "Leftenant",
        "gen.":  "Jeneral",
        "bro":   "broda",
        "bros":  "broda",
        "sis":   "sista",
        "govt":  "goment",
        "govt.": "goment",
        "no.":   "numba",
        "nig.":  "naija",
        "etc.":  "and so on",
        "pls":   "please",
        "plz":   "please",
        "btw":   "by the way",
        "fyi":   "for your information",
        "lol":   "laugh",
        "smh":   "shake my head",
        "imo":   "in my opinion",
    },
}

# ── Digit-to-spoken-word tables ────────────────────────────────────────────────
DIGITS = {
    "yo": {
        "0": "òdo",      "1": "ọ̀kan",   "2": "éjì",
        "3": "ẹ́ta",     "4": "ẹ́rin",   "5": "árùn-ún",
        "6": "ẹ́fà",     "7": "éje",     "8": "ẹ́jọ",
        "9": "ẹ́sàn-án",
    },
    "ig": {
        "0": "efu",      "1": "otu",     "2": "abụo",
        "3": "atọ",      "4": "anọ",     "5": "ise",
        "6": "isi",      "7": "asaa",    "8": "asatọ",
        "9": "itoolu",
    },
    "pcm": {
        "0": "zero",     "1": "one",     "2": "two",
        "3": "three",    "4": "four",    "5": "five",
        "6": "six",      "7": "seven",   "8": "eight",
        "9": "nine",
    },
}


class TTSNormalizer:
    """
    Normalizes text for TTS processing by expanding numbers, abbreviations, and
    symbols into their spoken equivalents.

    Args:
        lang: Target language. One of ``'yo'`` (Yoruba), ``'ig'`` (Igbo),
              ``'pcm'`` (Nigerian Pidgin). Defaults to ``'yo'``.
    """

    def __init__(self, lang: str = "yo"):
        self.lang = lang.lower()
        self.abbrev_map = ABBREVIATIONS.get(self.lang, {})
        self.digit_map = DIGITS.get(self.lang, {})

        if self.abbrev_map:
            escaped = [re.escape(k) for k in sorted(self.abbrev_map, key=len, reverse=True)]
            self.abbrev_re = re.compile(
                r'\b(?:' + '|'.join(escaped) + r')(?!\w)',
                flags=re.IGNORECASE,
            )
        else:
            self.abbrev_re = None

    def _replace_abbreviations(self, match: re.Match) -> str:
        return self.abbrev_map.get(match.group(0).lower(), match.group(0))

    def expand_abbreviations(self, text: str) -> str:
        """Expand abbreviations to their spoken forms."""
        if not text or not self.abbrev_re:
            return text
        return self.abbrev_re.sub(self._replace_abbreviations, text)

    def expand_numbers(self, text: str) -> str:
        """Expand digit characters to spoken words (digit-by-digit)."""
        if not text or not self.digit_map:
            return text

        def _replace(m: re.Match) -> str:
            return " " + self.digit_map.get(m.group(0), m.group(0)) + " "

        return re.sub(r'\s+', ' ', re.sub(r'\d', _replace, text)).strip()

    def normalize(self, text: str) -> str:
        """
        Run the full normalization pipeline: abbreviations → numbers.

        Args:
            text: Raw input text.

        Returns:
            Normalized text ready for phonetic processing.
        """
        if not text:
            return ""
        text = self.expand_abbreviations(text)
        text = self.expand_numbers(text)
        return text


class NaijaNormalizer(TTSNormalizer):
    """
    Extended text normalizer for **Nigerian Pidgin English** (Naija / ``pcm``).

    Inherits the full ``TTSNormalizer`` pipeline and adds Pidgin-specific
    informal spelling normalization — collapsing common alternate spellings
    to a canonical spoken form before TTS processing.

    Args:
        canonical: If ``True`` (default), apply informal-spelling normalization
                   before the standard abbreviation + number pipeline.
                   Set to ``False`` to use only the base ``TTSNormalizer`` behaviour.

    Example::

        from olaverse.nlp import NaijaNormalizer

        norm = NaijaNormalizer()
        norm.normalize("Oga, e don finish. Call am 2moro.")
        # → 'Oga, e don finish. Call am tomorrow.'
    """

    # Common Pidgin informal spellings → canonical form for TTS
    _INFORMAL: dict = {
        "2moro":  "tomorrow",
        "2day":   "today",
        "2nite":  "tonight",
        "4ward":  "forward",
        "b4":     "before",
        "luv":    "love",
        "hav":    "have",
        "wiv":    "with",
        "dis":    "this",
        "dat":    "that",
        "dem":    "them",
        "dey":    "they",
        "d":      "the",
        "u":      "you",
        "ur":     "your",
        "r":      "are",
        "nd":     "and",
        "n":      "and",
        "ok":     "okay",
        "kk":     "okay",
        "cuz":    "because",
        "cos":    "because",
        "tho":    "though",
        "thru":   "through",
        "nite":   "night",
        "morn":   "morning",
        "evri":   "every",
        "hw":     "how",
        "whr":    "where",
        "wen":    "when",
        "wot":    "what",
        "hv":     "have",
        "nt":     "not",
        "dnt":    "don't",
        "cnt":    "can't",
        "wnt":    "won't",
        "shd":    "should",
        "cld":    "could",
        "wld":    "would",
        "pls":    "please",
        "plz":    "please",
        "thnks":  "thanks",
        "tnx":    "thanks",
        "lol":    "laugh",
        "omg":    "oh my God",
        "smh":    "shake my head",
        "tbh":    "to be honest",
    }

    def __init__(self, canonical: bool = True):
        super().__init__(lang="pcm")
        self.canonical = canonical

        if self._INFORMAL:
            escaped = [re.escape(k) for k in sorted(self._INFORMAL, key=len, reverse=True)]
            self._informal_re = re.compile(
                r'\b(?:' + '|'.join(escaped) + r')\b',
                flags=re.IGNORECASE,
            )
        else:
            self._informal_re = None

    def _replace_informal(self, match: re.Match) -> str:
        return self._INFORMAL.get(match.group(0).lower(), match.group(0))

    def normalize_informal(self, text: str) -> str:
        """Collapse Pidgin informal spellings to canonical spoken forms."""
        if not text or not self._informal_re:
            return text
        return self._informal_re.sub(self._replace_informal, text)

    def normalize(self, text: str) -> str:
        """
        Full normalization pipeline for Pidgin:
        informal spellings → abbreviations → numbers.

        Args:
            text: Raw Pidgin input text.

        Returns:
            TTS-ready normalized text.
        """
        if not text:
            return ""
        if self.canonical:
            text = self.normalize_informal(text)
        text = self.expand_abbreviations(text)
        text = self.expand_numbers(text)
        return text
