"""
Unicode operations for diacritic tagging.

The central idea: every diacritized character is factorized into three parts

    character  =  base  +  SHAPE  +  TONE

where SHAPE is the set of marks that change *letter identity* (Yoruba underdot,
Vietnamese horn/circumflex/breve, Polish ogonek/stroke, Turkish dotless-i,
Hausa hooks, Spanish tilde-n ...) and TONE is the single mark that changes
*pitch or stress* (acute, grave, macron, and in Vietnamese also hook / tilde /
dot-below).

Whether a given combining mark is SHAPE or TONE is language-dependent and that
is the whole point:

    U+0323 dot below   ->  SHAPE in Yoruba (e vs e-underdot are different letters)
                       ->  TONE  in Vietnamese (nang tone)
    U+0303 tilde       ->  SHAPE in Spanish/Portuguese (n-tilde, a-tilde nasal)
                       ->  TONE  in Vietnamese (nga tone)

Factorizing this way collapses Vietnamese's ~60 flat classes into 5 shapes x 6
tones composed from parts that are each seen thousands of times, and it gives
the tone-direction errors (the dominant Yoruba failure mode) their own head,
their own loss term, their own confidence score and their own metric.

Everything here is pure Python + unicodedata. No model dependencies. The label
space of a trained checkpoint is defined by this file, so it is versioned:
changing SPEC or SPECIAL_LETTERS invalidates existing checkpoints.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

SPEC_VERSION = "1.2.0"

# --------------------------------------------------------------------------
# combining marks we care about
# --------------------------------------------------------------------------

ACUTE = "\u0301"
GRAVE = "\u0300"
CIRCUMFLEX = "\u0302"
TILDE = "\u0303"
MACRON = "\u0304"
BREVE = "\u0306"
DOT_ABOVE = "\u0307"
DIAERESIS = "\u0308"
HOOK_ABOVE = "\u0309"
DOUBLE_ACUTE = "\u030b"
CARON = "\u030c"
HORN = "\u031b"
DOT_BELOW = "\u0323"
CEDILLA = "\u0327"
OGONEK = "\u0328"

MARK_NAMES = {
    ACUTE: "ACUTE",
    GRAVE: "GRAVE",
    CIRCUMFLEX: "CIRCUMFLEX",
    TILDE: "TILDE",
    MACRON: "MACRON",
    BREVE: "BREVE",
    DOT_ABOVE: "DOT_ABOVE",
    DIAERESIS: "DIAERESIS",
    HOOK_ABOVE: "HOOK_ABOVE",
    DOUBLE_ACUTE: "DOUBLE_ACUTE",
    CARON: "CARON",
    HORN: "HORN",
    DOT_BELOW: "DOT_BELOW",
    CEDILLA: "CEDILLA",
    OGONEK: "OGONEK",
}

# The complete set of combining marks this model handles. A codepoint that is
# not in here is not a diacritic *for this task*, even if Unicode calls it one.
#
# This whitelist is load-bearing. Without it, NFD-decomposing arbitrary text
# invents label classes out of any script that happens to appear in the corpus:
# Hausa Wikipedia carries Ajami (Arabic), Yoruba Wikipedia carries Korean and
# Devanagari names, and a Hangul syllable decomposes into three combining-class-0
# jamo that look exactly like "base + two marks" to a naive splitter.
HANDLED_MARKS: FrozenSet[str] = frozenset(MARK_NAMES)


@lru_cache(maxsize=1 << 16)
def _is_latin_cp(ch: str) -> bool:
    try:
        return unicodedata.name(ch).startswith("LATIN")
    except ValueError:
        return False


def _split_marks(ch: str) -> Tuple[str, List[str]]:
    """NFD-split one grapheme into (base, marks), or treat it as atomic.

    A trailing codepoint counts as a mark only if it is a real combining mark
    (combining class != 0) *and* one we model. Anything else means the character
    is outside this model's alphabet: it is returned whole, so it round-trips
    untouched and contributes only the identity class.
    """
    d = unicodedata.normalize("NFD", ch)
    base, rest = d[0], d[1:]
    if not rest:
        return base, []
    # The base must be Latin. Greek and Cyrillic share the combining acute
    # (U+0301) with Latin, so a whitelist of marks alone is not enough --
    # without this check, Greek text passed through the model would come back
    # with its accents removed.
    if not (base.isascii() or _is_latin_cp(base)):
        return ch, []
    for m in rest:
        if unicodedata.combining(m) == 0 or m not in HANDLED_MARKS:
            return ch, []
    return base, list(rest)


# --------------------------------------------------------------------------
# letters that do not decompose under NFD and must be handled by substitution
# --------------------------------------------------------------------------
# maps  special letter -> (ascii base, shape tag)
SPECIAL_LETTERS: Dict[str, Tuple[str, str]] = {
    "\u0142": ("l", "STROKE"),      # l  Polish
    "\u0141": ("L", "STROKE"),      # L
    "\u0111": ("d", "STROKE"),      # d  Vietnamese
    "\u0110": ("D", "STROKE"),      # D
    "\u0131": ("i", "DOTLESS"),     # dotless i  Turkish
    "\u0253": ("b", "HOOK"),        # b-hook  Hausa
    "\u0181": ("B", "HOOK"),
    "\u0257": ("d", "HOOK"),        # d-hook  Hausa
    "\u018a": ("D", "HOOK"),
    "\u0199": ("k", "HOOK"),        # k-hook  Hausa
    "\u0198": ("K", "HOOK"),
    "\u01b4": ("y", "HOOK"),        # y-hook  Hausa
    "\u01b3": ("Y", "HOOK"),
    "\u0272": ("n", "LEFTHOOK"),
    "\u019d": ("N", "LEFTHOOK"),
}

LANGS: Tuple[str, ...] = (
    "yor", "ibo", "hau", "vie", "pol", "tur", "por", "spa", "fra", "ita",
)

# ISO-639-1 -> ISO-639-3, for datasets that use two-letter codes
LANG_ALIASES: Dict[str, str] = {
    "yo": "yor", "ig": "ibo", "ha": "hau", "vi": "vie", "pl": "pol",
    "tr": "tur", "pt": "por", "es": "spa", "fr": "fra", "it": "ita",
}


@dataclass(frozen=True)
class LangSpec:
    """Per-language declaration of the writing system.

    `all_marks` and `special_tags` together define the language's alphabet.
    A character carrying anything outside them is not a character of this
    language -- it is a foreign word that happens to share the Latin script.
    """
    code: str
    tone_marks: FrozenSet[str]
    all_marks: FrozenSet[str]
    # the actual non-decomposing letters this language uses. Tags alone are
    # ambiguous: Polish l-stroke and Vietnamese d-stroke share the tag STROKE.
    special_letters: FrozenSet[str] = frozenset()
    notes: str = ""


def _fs(*marks: str) -> FrozenSet[str]:
    return frozenset(marks)


SPEC: Dict[str, LangSpec] = {
    "yor": LangSpec(
        "yor",
        tone_marks=_fs(ACUTE, GRAVE, MACRON),
        all_marks=_fs(ACUTE, GRAVE, MACRON, DOT_BELOW),
        notes="dot-below marks letter identity (e/o/s), acute+grave mark tone",
    ),
    "ibo": LangSpec(
        "ibo",
        tone_marks=_fs(ACUTE, GRAVE, MACRON),
        all_marks=_fs(ACUTE, GRAVE, MACRON, DOT_BELOW, DOT_ABOVE),
        notes="dot-below on i/u/o, dot-above on n; acute+grave mark tone",
    ),
    "hau": LangSpec(
        "hau",
        special_letters=_fs("\u0253","\u0181","\u0257","\u018a","\u0199","\u0198","\u01b4","\u01b3","\u0272","\u019d"),
        tone_marks=_fs(ACUTE, GRAVE, MACRON, CIRCUMFLEX),
        all_marks=_fs(ACUTE, GRAVE, MACRON, CIRCUMFLEX),
        notes="hooked letters are non-decomposing specials; tone rarely written",
    ),
    "vie": LangSpec(
        "vie",
        special_letters=_fs("\u0111","\u0110"),
        tone_marks=_fs(ACUTE, GRAVE, HOOK_ABOVE, TILDE, DOT_BELOW),
        all_marks=_fs(ACUTE, GRAVE, HOOK_ABOVE, TILDE, DOT_BELOW,
                      CIRCUMFLEX, BREVE, HORN),
        notes="circumflex/breve/horn are vowel quality (shape); 5 written tones",
    ),
    "pol": LangSpec(
        "pol",
        special_letters=_fs("\u0142","\u0141"),
        tone_marks=_fs(ACUTE),
        all_marks=_fs(ACUTE, OGONEK, DOT_ABOVE),
        notes="kreska treated as tone slot; ogonek/kropka/stroke are shape",
    ),
    "tur": LangSpec(
        "tur",
        special_letters=_fs("\u0131"),
        tone_marks=frozenset(),
        all_marks=_fs(CEDILLA, BREVE, DIAERESIS, CIRCUMFLEX, DOT_ABOVE),
        notes="no tone slot; dotless-i is a shape special",
    ),
    "por": LangSpec(
        "por",
        tone_marks=_fs(ACUTE, GRAVE),
        all_marks=_fs(ACUTE, GRAVE, CIRCUMFLEX, TILDE, CEDILLA, DIAERESIS),
        notes="tilde is nasal quality (shape), not tone",
    ),
    "spa": LangSpec(
        "spa",
        tone_marks=_fs(ACUTE),
        all_marks=_fs(ACUTE, TILDE, DIAERESIS),
        notes="tilde-n is a letter (shape)",
    ),
    "fra": LangSpec(
        "fra",
        tone_marks=_fs(ACUTE, GRAVE),
        all_marks=_fs(ACUTE, GRAVE, CIRCUMFLEX, DIAERESIS, CEDILLA),
    ),
    "ita": LangSpec(
        "ita",
        tone_marks=_fs(ACUTE, GRAVE),
        all_marks=_fs(ACUTE, GRAVE, CIRCUMFLEX),
    ),
}

# a mark that is in no language's tone slot is always shape
ALL_TONE_MARKS: FrozenSet[str] = frozenset().union(*(s.tone_marks for s in SPEC.values()))


def normalize_lang(code: Optional[str]) -> Optional[str]:
    if code is None:
        return None
    c = code.strip().lower().replace("-", "_").split("_")[0]
    c = LANG_ALIASES.get(c, c)
    return c if c in SPEC else None


# --------------------------------------------------------------------------
# factorization
# --------------------------------------------------------------------------

Shape = Tuple[str, ...]   # e.g. ()  ("DOT_BELOW",)  ("SPECIAL:STROKE",)
Tone = str                # "" or a single combining mark


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def graphemes(text: str) -> List[str]:
    """Split into user-perceived characters: a base codepoint plus any
    combining marks that follow it.

    This matters more than it looks.  NFC does *not* give one codepoint per
    character: Yoruba stacked marks have no precomposed form, so NFC of
    'gbo-underdot-acute' is two codepoints, and naive iteration over
    `str` misaligns every label after that point and breaks the strip
    invariant.  The tagger's positions are graphemes, always.
    """
    out: List[str] = []
    for c in nfc(text):
        if out and unicodedata.combining(c):
            out[-1] += c
        else:
            out.append(c)
    return out


def factorize_char(ch: str, lang: str) -> Tuple[str, Shape, Tone]:
    """Split one character into (base, shape, tone) for a given language.

    `base` is always a single ASCII-ish character with no marks.
    Round-trips exactly: compose_char(*factorize_char(c, l)) == nfc(c)
    """
    spec = SPEC[lang]
    if ch in SPECIAL_LETTERS:
        base, tag = SPECIAL_LETTERS[ch]
        return base, (f"SPECIAL:{tag}",), ""

    base, marks = _split_marks(ch)
    if base == ch and marks == [] and len(ch) > 1:
        return ch, (), ""          # atomic, outside our alphabet

    # a special letter can itself carry combining marks, e.g. Hausa k-hook +
    # acute, or Polish l-stroke.  NFD leaves the special letter intact.
    shape_tags: List[str] = []
    if base in SPECIAL_LETTERS:
        b2, tag = SPECIAL_LETTERS[base]
        base = b2
        shape_tags.append(f"SPECIAL:{tag}")

    tone: Tone = ""
    for m in marks:
        if m in spec.tone_marks and not tone:
            tone = m
        else:
            shape_tags.append(MARK_NAMES.get(m, f"U+{ord(m):04X}"))
    return base, tuple(shape_tags), tone


_TAG_TO_MARK = {v: k for k, v in MARK_NAMES.items()}
_SPECIAL_BY_TAG: Dict[Tuple[str, str], str] = {
    (base, tag): letter for letter, (base, tag) in SPECIAL_LETTERS.items()
}


def compose_char(base: str, shape: Shape, tone: Tone) -> str:
    """Inverse of factorize_char. Order matters: shape marks precede the tone
    mark so that Vietnamese circumflex+acute composes to the right codepoint
    (both have combining class 230, so NFC will not reorder them for us)."""
    out = base
    marks: List[str] = []
    for tag in shape:
        if tag.startswith("SPECIAL:"):
            key = (out, tag.split(":", 1)[1])
            if key not in _SPECIAL_BY_TAG:
                return ""          # illegal combination
            out = _SPECIAL_BY_TAG[key]
        else:
            m = _TAG_TO_MARK.get(tag)
            if m is None:
                return ""
            marks.append(m)
    if tone:
        marks.append(tone)
    return nfc(out + "".join(marks))


def base_char(ch: str) -> str:
    """Language-independent base form of a grapheme = what a user types on an
    ASCII keyboard.  This is the strip() that defines the task, and it is the
    invariant the tagger is structurally unable to violate."""
    if ch in SPECIAL_LETTERS:
        return SPECIAL_LETTERS[ch][0]
    b, marks = _split_marks(ch)
    if b == ch and not marks:
        return ch                  # atomic: strip is a no-op, as it must be
    if b in SPECIAL_LETTERS:
        b = SPECIAL_LETTERS[b][0]
    return b


def strip_diacritics(text: str) -> str:
    """Remove every diacritic. One grapheme in, one base character out."""
    return "".join(base_char(g) for g in graphemes(text))


def partial_strip(text: str, keep_prob: float, rng) -> str:
    """Remove diacritics from a random subset of characters.

    Real business inputs are rarely fully bare: people type some accents and
    skip others, or paste text that was half-corrected.  Training only on
    fully-stripped input makes the model brittle on those, so we train on the
    whole spectrum from bare to fully marked.
    """
    if keep_prob <= 0.0:
        return strip_diacritics(text)
    out = []
    for g in graphemes(text):
        b = base_char(g)
        out.append(g if (b != g and rng.random() < keep_prob) else b)
    return "".join(out)


# --------------------------------------------------------------------------
# density: the quality gate for training data
# --------------------------------------------------------------------------

def diacritic_density(text: str, lang: str) -> float:
    """Fraction of diacritic-eligible characters that actually carry a mark.

    'Eligible' is approximated by 'is a cased letter'. Matches the definition
    used to build diacnet-1.1-corpus closely enough to reuse its thresholds.
    """
    eligible = 0
    marked = 0
    for g in graphemes(text):
        if not g[0].isalpha():
            continue
        eligible += 1
        if base_char(g) != g:
            marked += 1
    return (marked / eligible) if eligible else 0.0


def is_eligible(base: str) -> bool:
    return base.isalpha()


# --------------------------------------------------------------------------
# script filter
# --------------------------------------------------------------------------

def latin_fraction(text: str) -> float:
    """Fraction of alphabetic characters written in Latin script.

    All ten target languages use the Latin alphabet. Sentences that do not are
    not merely useless as training data -- they are actively harmful, because
    they contribute nothing but identity labels and skew the class balance.
    """
    letters = [c for c in nfc(text) if c.isalpha()]
    if not letters:
        return 1.0
    return sum(_is_latin_cp(c) for c in letters) / len(letters)


def is_latin_text(text: str, threshold: float = 0.9) -> bool:
    return latin_fraction(text) >= threshold


# --------------------------------------------------------------------------
# per-language alphabet check
# --------------------------------------------------------------------------

@lru_cache(maxsize=1 << 18)
def in_inventory(g: str, lang: str) -> bool:
    """Is this grapheme a character of `lang`?

    The script filter cannot catch cross-language contamination, because every
    target language is Latin: a Vietnamese name inside a Turkish sentence, or a
    Czech word inside a Polish one, passes it cleanly. But `ệ` read as Turkish
    factorizes to shape ('DOT_BELOW','CIRCUMFLEX') -- Turkish declares no tone
    slot, so both marks land in the shape head and invent a composite class
    that exists nowhere in Turkish.
    """
    spec = SPEC.get(lang)
    if spec is None:
        return False
    if g in SPECIAL_LETTERS:
        return g in spec.special_letters
    base, marks = _split_marks(g)
    if base == g and not marks:
        return True                          # plain or atomic: harmless
    if base in SPECIAL_LETTERS and base not in spec.special_letters:
        return False
    if len(set(marks)) != len(marks):
        return False                         # e.g. two dots below: malformed
    return all(m in spec.all_marks for m in marks)


def foreign_grapheme_count(text: str, lang: str) -> int:
    return sum(not in_inventory(g, lang) for g in graphemes(text))


def is_clean_for_lang(text: str, lang: str, max_foreign: int = 0) -> bool:
    """Sentence-level gate: no characters from other languages' alphabets."""
    return foreign_grapheme_count(text, lang) <= max_foreign


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

def factorize_text(text: str, lang: str) -> Tuple[List[str], List[Shape], List[Tone]]:
    """Factorize a whole string. Returns three lists, all the same length as
    graphemes(text): base characters, shapes, tones."""
    bases: List[str] = []
    shapes: List[Shape] = []
    tones: List[Tone] = []
    for g in graphemes(text):
        b, sh, tn = factorize_char(g, lang)
        bases.append(b)
        shapes.append(sh)
        tones.append(tn)
    return bases, shapes, tones


def compose_text(bases: Sequence[str], shapes: Sequence[Shape],
                 tones: Sequence[Tone]) -> str:
    return "".join(compose_char(b, s, t) or b
                   for b, s, t in zip(bases, shapes, tones))


def roundtrip_ok(text: str, lang: str) -> bool:
    for g in graphemes(text):
        b, s, t = factorize_char(g, lang)
        if compose_char(b, s, t) != g:
            return False
    return True
