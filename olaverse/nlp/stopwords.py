"""
Stopword lists for Nigerian languages.

Each set contains high-frequency function words — pronouns, prepositions,
conjunctions, auxiliaries, particles — that carry little semantic weight and
are typically filtered before tokenization, TF-IDF, or search indexing.

Usage::

    from olaverse.nlp.stopwords import YORUBA_STOPWORDS, IGBO_STOPWORDS
    from olaverse.nlp.stopwords import HAUSA_STOPWORDS, PIDGIN_STOPWORDS

    tokens = ["bawo", "ni", "Ade", "ṣe", "dara"]
    filtered = [t for t in tokens if t.lower() not in YORUBA_STOPWORDS]
    # → ['Ade', 'dara']
"""

YORUBA_STOPWORDS: frozenset = frozenset({
    # ── Personal pronouns ─────────────────────────────────────────────
    "mo", "mi", "o", "ọ", "a", "ẹ", "wọn", "rẹ", "wa",
    # ── Possessives ───────────────────────────────────────────────────
    "temi", "tirẹ", "tire", "tiwa", "tiwọn",
    # ── Prepositions ──────────────────────────────────────────────────
    "si", "sí", "ni", "ní", "fun", "fún", "lati", "láti",
    "pẹlu", "pẹ̀lú", "bi", "bí", "ninu", "nínú", "lara", "láàárín",
    "lori", "lórí", "abẹ", "agbegbe",
    # ── Conjunctions ──────────────────────────────────────────────────
    "ati", "àti", "tabi", "tàbí", "sugbon", "ṣugbọn",
    "nitori", "nítorí", "bẹẹ", "bẹ̀ẹ̀", "pe", "pé", "ti", "tí",
    "nigbati", "nígbàtí", "bi", "bí",
    # ── Common function verbs ─────────────────────────────────────────
    "je", "jẹ", "wa", "wà", "ṣe", "se", "lo", "lọ", "wá",
    "fi", "de", "dé", "ba", "bá", "ko", "kò", "ri", "gbọ", "kan",
    # ── Determiners / demonstratives ─────────────────────────────────
    "naa", "náà", "yii", "yìí", "yen", "yẹn", "na", "ná", "yi", "yí",
    # ── Question words ────────────────────────────────────────────────
    "bawo", "ibo", "ibọ", "kini", "tani", "nigba", "nigbati", "nibo",
    "ibo", "melo", "mẹ́lo",
    # ── Negation ──────────────────────────────────────────────────────
    "ko", "kò", "ki", "kí", "kii", "kìí",
    # ── Particles ─────────────────────────────────────────────────────
    "n", "k", "ni", "ní",
    # ── Common adverbs ────────────────────────────────────────────────
    "paapaa", "pàápàá", "gaan", "gidi", "lasan",
})

IGBO_STOPWORDS: frozenset = frozenset({
    # ── Personal pronouns ─────────────────────────────────────────────
    "m", "mu", "gi", "gị", "ya", "anyi", "anyị", "unu", "ha",
    "o", "ọ", "ha", "uche",
    # ── Prepositions ──────────────────────────────────────────────────
    "na", "n", "ime", "elu", "okpuru",
    "site", "n'ihi", "maka", "n'ime", "n'elu", "n'okpuru",
    "n'elu", "n'okpuru", "n'ebe",
    # ── Conjunctions ──────────────────────────────────────────────────
    "ma", "mana", "ka", "iji", "naanị", "obula",
    "ọ bụ", "ma ọ bụ", "ọ bụrụ",
    # ── Auxiliaries / copula ──────────────────────────────────────────
    "bu", "bụ", "di", "dị", "no", "nọ", "ga", "na",
    "bụrụ", "dịrị",
    # ── Question words ────────────────────────────────────────────────
    "gini", "onye", "ebe", "mgbe", "olee", "ole", "kedu", "olee kwanye",
    # ── Determiners ───────────────────────────────────────────────────
    "a", "ya", "nke", "ahụ", "a",
    # ── Negation ──────────────────────────────────────────────────────
    "abughi", "abụghị", "ọ bụghị", "e cheghị",
    # ── Particles / affixes ───────────────────────────────────────────
    "e", "i", "ọ", "a", "u", "ụ",
    # ── Common adverbs ────────────────────────────────────────────────
    "ọzọ", "ozizi", "taa", "ụbọchị", "ugbu a",
})

HAUSA_STOPWORDS: frozenset = frozenset({
    # ── Personal pronouns ─────────────────────────────────────────────
    "ni", "kai", "ke", "shi", "ita", "mu", "ku", "su", "an",
    "na", "ta", "ya", "mun", "kun", "sun",
    # ── Prepositions ──────────────────────────────────────────────────
    "a", "ba", "da", "daga", "ga", "gare", "kan",
    "karkashin", "bayan", "gaban", "tsakanin", "wajen",
    "tare", "tare da", "cikin",
    # ── Conjunctions ──────────────────────────────────────────────────
    "da", "ko", "ko da", "amma", "amma kuwa",
    "don", "domin", "saboda", "sai", "kuma", "tunda",
    # ── Tense markers / auxiliaries ───────────────────────────────────
    "ne", "ce", "nake", "kake", "yake", "tana", "muna", "kuna", "suna",
    "zan", "na", "ka", "ya", "ta", "mun", "kun", "sun",
    # ── Question words ────────────────────────────────────────────────
    "me", "wane", "wacce", "ina", "yaya", "yaushe", "nawa", "daga ina",
    "wa",
    # ── Negation ──────────────────────────────────────────────────────
    "ba", "bai", "bata", "ba mu", "ba ku", "ba su",
    # ── Particles ─────────────────────────────────────────────────────
    "ne", "ce", "ke nan", "fa", "kuwa",
    # ── Determiners ───────────────────────────────────────────────────
    "wannan", "wancan", "waɗannan", "waɗancan",
    # ── Common adverbs ────────────────────────────────────────────────
    "yanzu", "koyaushe", "kadan", "kaɗan", "sosai",
})

PIDGIN_STOPWORDS: frozenset = frozenset({
    # ── Personal pronouns ─────────────────────────────────────────────
    "i", "you", "yu", "him", "am", "she", "her",
    "we", "wi", "una", "dem", "im", "e",
    # ── Auxiliaries / function words ──────────────────────────────────
    "dey", "de", "na", "be", "go", "don", "fit", "get", "make",
    "com", "come", "wey", "sey", "say", "sabi", "sef",
    # ── Prepositions ──────────────────────────────────────────────────
    "for", "wit", "with", "from", "to", "of",
    "for inside", "for top", "for bottom", "for back",
    # ── Conjunctions ──────────────────────────────────────────────────
    "and", "but", "so", "because", "cos", "if", "when",
    "afta", "before", "as", "or", "tho", "though",
    # ── Articles / determiners ────────────────────────────────────────
    "di", "de", "the", "a", "an", "one", "dat", "dis",
    "that", "this", "those", "these", "some",
    # ── Common function verbs ─────────────────────────────────────────
    "do", "bin", "wan", "go", "fit", "take", "put", "give",
    # ── Negation ──────────────────────────────────────────────────────
    "no", "neva", "never", "not", "nor",
    # ── Question words ────────────────────────────────────────────────
    "wetin", "who", "how", "where", "when", "which", "why",
    "wia", "who", "wetin",
    # ── Particles ─────────────────────────────────────────────────────
    "o", "oh", "na", "sha", "self", "sef", "even",
    # ── Common adverbs ────────────────────────────────────────────────
    "well", "now", "still", "already", "just", "then",
    "again", "sef", "too", "very", "so", "small", "small small",
})


def get_stopwords(lang: str) -> frozenset:
    """
    Return the stopword set for a language code.

    Args:
        lang: ISO language code — ``'yor'``, ``'ibo'``, ``'hau'``, ``'pcm'``, or ``'eng'``.
              Also accepts short codes ``'yo'``, ``'ig'``, ``'ha'``.
              English returns an empty set (use NLTK or spaCy for English stopwords).

    Returns:
        frozenset of stopword strings (lowercased).

    Raises:
        ValueError: If the language code is not recognised.
    """
    _MAP = {
        "yor": YORUBA_STOPWORDS, "yo": YORUBA_STOPWORDS,
        "ibo": IGBO_STOPWORDS,   "ig": IGBO_STOPWORDS,
        "hau": HAUSA_STOPWORDS,  "ha": HAUSA_STOPWORDS,
        "pcm": PIDGIN_STOPWORDS,
        "eng": frozenset(),
    }
    code = lang.lower().strip()
    if code not in _MAP:
        raise ValueError(
            f"Language '{lang}' not recognised. "
            f"Available: 'yor', 'ibo', 'hau', 'pcm', 'eng'."
        )
    return _MAP[code]


def filter_stopwords(tokens: list, lang: str) -> list:
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens: List of string tokens (as returned by ``Tokenizer.decode`` or ``str.split``).
        lang: Language code — ``'yor'``, ``'ibo'``, ``'hau'``, ``'pcm'``.

    Returns:
        List of tokens with stopwords removed (case-insensitive match).

    Example::

        from olaverse.nlp.stopwords import filter_stopwords

        tokens = ["bawo", "ni", "Ade", "ṣe", "dara", "?"]
        filter_stopwords(tokens, "yor")
        # → ['Ade', 'dara', '?']
    """
    sw = get_stopwords(lang)
    return [t for t in tokens if t.lower() not in sw]
