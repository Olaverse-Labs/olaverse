"""
Helpers for the diacnet-2.0 family (text-to-text diacritizers).

These are model-free and importable without torch or transformers:

* :func:`align`       keep the input's letters, take only the model's marks
* :func:`strip_marks` the base-letter skeleton that :func:`align` preserves
* :func:`chunk_text`  split long text into ~300-character pieces on spaces
* :func:`build_hint`  format ``[g: word=meaning]`` meaning hints
* :func:`resolve_tag` language code -> the model's control tag

``align`` is the reference function published on the diacnet-2.0 model card; it
is verified identical to the benchmark scorer.
"""

import difflib
import unicodedata as ud
from collections.abc import Mapping
from typing import Iterable, List, Optional, Union

__all__ = [
    "DIACNET2_TAGS",
    "align",
    "build_hint",
    "chunk_text",
    "resolve_tag",
    "strip_marks",
]

_FOLD = str.maketrans("ɓɗƙƴđıłƁƊƘƳĐŁ", "bdkydilBDKYDL")
_LETTER = {"ٓ", "ٔ", "ٕ"}      # Arabic madda / hamza are spelling, not marks


def _units(text):
    units = []
    for c in ud.normalize("NFD", text):
        if units and ud.combining(c):
            if c in _LETTER:
                units[-1][0] += c
            units[-1][1] += c
        else:
            units.append([c, c])
    return [(ud.normalize("NFC", b).translate(_FOLD), ud.normalize("NFC", f)) for b, f in units]


def align(source, output):
    """Return ``source`` with the marks the model put on every letter it kept.

    Letters the model changed, dropped or added fall back to ``source``, so the
    text itself never changes: ``strip_marks(align(s, o)) == strip_marks(s)``.
    """
    src, out = _units(source), _units(output)
    res = [f for _, f in src]
    sm = difflib.SequenceMatcher(None, [b for b, _ in src], [b for b, _ in out], autojunk=False)
    for a, b, n in sm.get_matching_blocks():
        res[a:a + n] = [f for _, f in out[b:b + n]]
    return "".join(res)


def strip_marks(text: str) -> str:
    """The text's base letters: every mark removed, hooked and barred letters
    (ɓ ɗ ƙ ƴ đ ı ł) folded to their plain form, Arabic hamza/madda kept.

    This is the skeleton :func:`align` leaves untouched.
    """
    return "".join(b for b, _ in _units(text))


def chunk_text(text: str, n: int = 300) -> List[str]:
    """Split ``text`` into pieces of at most about ``n`` characters, on spaces.

    Greedy: words are packed until the next one would overflow. A single word
    longer than ``n`` is kept whole rather than cut. Empty input gives ``[]``.
    The pieces rejoin with a single space.
    """
    if not text:
        return []
    out, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > n:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    return out + [cur]


# --------------------------------------------------------------------------
# language tags
# --------------------------------------------------------------------------

#: Control tags diacnet-2.0 and diacnet-mini-2.0 were trained with.
DIACNET2_TAGS = ("yor", "ibo", "hau", "vie", "pol", "tur", "por", "spa",
                 "fra", "ita", "ara", "ara-nocase", "auto")

_ISO1 = {
    "yo": "yor", "ig": "ibo", "ha": "hau", "vi": "vie", "pl": "pol",
    "tr": "tur", "pt": "por", "es": "spa", "fr": "fra", "it": "ita",
    "ar": "ara",
}


def resolve_tag(lang: Optional[str], case_endings: bool = True) -> str:
    """Map a language code to the model's control tag (without the angle brackets).

    Accepts ISO-639-3 (``"yor"``), ISO-639-1 (``"yo"``), the tags ``"auto"`` and
    ``"ara-nocase"``, and ``None`` (meaning ``"auto"``). With
    ``case_endings=False``, Arabic resolves to ``"ara-nocase"``; the switch has
    no effect on any other language.

    Raises:
        ValueError: for a language the model has no tag for.
    """
    if lang is None:
        return "auto"
    key = str(lang).strip().lower().strip("<>").replace("_", "-")
    key = _ISO1.get(key, key)
    if key not in DIACNET2_TAGS:
        raise ValueError(
            f"Unsupported language '{lang}' for diacnet-2.0. Supported: "
            f"{list(DIACNET2_TAGS)} (ISO-639-1 codes such as 'yo' are also "
            f"accepted). Pass lang=None for auto-detection."
        )
    if key == "ara" and not case_endings:
        return "ara-nocase"
    return key


# --------------------------------------------------------------------------
# meaning hints
# --------------------------------------------------------------------------

Hints = Union[None, str, Mapping, Iterable]


def _hint_pair(word, meaning) -> str:
    word, meaning = str(word).strip(), str(meaning).strip()
    if not word or not meaning:
        raise ValueError("A meaning hint needs both a word and a meaning: 'word=meaning'.")
    if "=" in word:
        raise ValueError(f"Hint word {word!r} must not contain '='.")
    for part in (word, meaning):
        if any(ch in part for ch in "|]\n"):
            raise ValueError(
                f"Hint text {part!r} must not contain '|', ']' or a newline; "
                f"they delimit the hint block."
            )
    return f"{word}={meaning}"


def build_hint(hints: Hints) -> str:
    """Format meaning hints as the ``[g: word=meaning | word=meaning]`` block.

    Accepts a mapping ``{word: meaning}``, a list of ``"word=meaning"`` strings
    or ``(word, meaning)`` pairs, a single ``"word=meaning"`` string (several may
    be separated by ``" | "``), or a ready-made ``"[g: ...]"`` block, which is
    passed through. ``None`` or an empty value gives ``""``.

    Hints are written in English and steer the choice of marks for words whose
    diacritics depend on meaning (Vietnamese *ranh*, Yorùbá *ogun*).

    Raises:
        ValueError: for a hint without a word or meaning, or containing the
            characters that delimit the block.
    """
    if hints is None:
        return ""
    if isinstance(hints, str):
        s = hints.strip()
        if not s:
            return ""
        if s.startswith("[g:"):
            return s
        items = [p for p in s.split("|")]
    elif isinstance(hints, Mapping):
        items = list(hints.items())
    else:
        items = list(hints)
    if not items:
        return ""

    pairs = []
    for item in items:
        if isinstance(item, str):
            if "=" not in item:
                raise ValueError(
                    f"Hint {item.strip()!r} must look like 'word=meaning'."
                )
            word, meaning = item.split("=", 1)
        else:
            try:
                word, meaning = item
            except (TypeError, ValueError):
                raise ValueError(
                    f"Hint {item!r} must be a 'word=meaning' string or a "
                    f"(word, meaning) pair."
                ) from None
        pairs.append(_hint_pair(word, meaning))
    return "[g: " + " | ".join(pairs) + "]"
