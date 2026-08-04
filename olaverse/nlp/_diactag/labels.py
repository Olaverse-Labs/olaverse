"""
Label space and legality masks.

The tagger has two classification heads per position:

    SHAPE head  ->  which letter-identity marks does this character carry
    TONE  head  ->  which tone/stress mark does it carry (or none)

Neither head can change the base character, which is where the structural
guarantee comes from: strip(output) == strip(input), by construction, always.

On top of that we add a *legality mask*: for each (language, base character)
we record which (shape, tone) combinations were ever observed in the training
corpus, and mask the rest to -inf at decode time.  That upgrades the guarantee
from "output has the right skeleton" to "every character the model emits is a
character that actually exists in this language" -- so no Polish l-stroke with
an acute on it, no Turkish tone marks, no Yoruba hook letters.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .unicode_ops import (
    LANGS, SPEC, SPEC_VERSION, Shape, Tone, base_char, factorize_char,
    graphemes, in_inventory, is_latin_text, nfc,
)

PAD, UNK, MASK, BOS, EOS = "<pad>", "<unk>", "<mask>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, MASK, BOS, EOS]
PAD_ID, UNK_ID, MASK_ID, BOS_ID, EOS_ID = range(5)


def _shape_key(s: Shape) -> str:
    return "+".join(s) if s else "NONE"


def _unshape_key(k: str) -> Shape:
    return () if k == "NONE" else tuple(k.split("+"))


@dataclass
class LabelSpace:
    """Everything needed to turn text into tensors and tensors back into text."""

    chars: List[str]                       # grapheme vocabulary (input side)
    shapes: List[str]                      # shape keys, index 0 == "NONE"
    tones: List[str]                       # tone marks, index 0 == ""
    langs: List[str]
    # legality[lang][base_char] -> set of (shape_idx, tone_idx)
    legal: Dict[str, Dict[str, List[Tuple[int, int]]]] = field(default_factory=dict)
    spec_version: str = SPEC_VERSION

    # ---- derived -------------------------------------------------------
    def __post_init__(self) -> None:
        self.char2id = {c: i for i, c in enumerate(self.chars)}
        self.shape2id = {s: i for i, s in enumerate(self.shapes)}
        self.tone2id = {t: i for i, t in enumerate(self.tones)}
        self.lang2id = {l: i for i, l in enumerate(self.langs)}

    @property
    def n_chars(self) -> int: return len(self.chars)

    @property
    def n_shapes(self) -> int: return len(self.shapes)

    @property
    def n_tones(self) -> int: return len(self.tones)

    @property
    def n_langs(self) -> int: return len(self.langs)

    # ---- encoding ------------------------------------------------------
    def encode(self, text_or_graphemes) -> List[int]:
        gs = (text_or_graphemes if isinstance(text_or_graphemes, list)
              else graphemes(text_or_graphemes))
        return [self.char2id.get(g, UNK_ID) for g in gs]

    def labels_for(self, gold: str, lang: str) -> Tuple[List[int], List[int]]:
        """Gold (shape, tone) label ids for a fully diacritized string."""
        sh, tn = [], []
        for g in graphemes(gold):
            b, s, t = factorize_char(g, lang)
            sh.append(self.shape2id.get(_shape_key(s), 0))
            tn.append(self.tone2id.get(t, 0))
        return sh, tn

    def decode_char(self, base: str, shape_id: int, tone_id: int) -> str:
        from .unicode_ops import compose_char
        out = compose_char(base, _unshape_key(self.shapes[shape_id]),
                           self.tones[tone_id])
        return out or base

    # ---- legality ------------------------------------------------------
    def legal_mask_array(self):
        """Dense bool array [n_langs, n_chars, n_shapes, n_tones].

        Indexed by the *input* grapheme id: a partially-marked input grapheme
        is legal for whatever its base character is legal for.  Tiny (a few
        hundred KB) so we just materialise it.
        """
        import numpy as np
        m = np.zeros((self.n_langs, self.n_chars, self.n_shapes, self.n_tones),
                     dtype=bool)
        for lang, li in self.lang2id.items():
            per_base = self.legal.get(lang, {})
            for ch, ci in self.char2id.items():
                if ci < len(SPECIAL_TOKENS):
                    m[li, ci, 0, 0] = True
                    continue
                b = base_char(ch)
                pairs = per_base.get(b)
                if not pairs:
                    m[li, ci, 0, 0] = True      # unseen base: identity only
                    continue
                for s_i, t_i in pairs:
                    m[li, ci, s_i, t_i] = True
        return m

    def eligible_mask_array(self):
        """[n_langs, n_chars] -> True if this character has >1 legal outcome.
        Used as the denominator for DER."""
        import numpy as np
        m = self.legal_mask_array()
        return m.reshape(self.n_langs, self.n_chars, -1).sum(-1) > 1

    # ---- io ------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "spec_version": self.spec_version,
                "chars": self.chars,
                "shapes": self.shapes,
                "tones": self.tones,
                "langs": self.langs,
                "legal": {l: {b: [list(p) for p in ps] for b, ps in d.items()}
                          for l, d in self.legal.items()},
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LabelSpace":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d["spec_version"] != SPEC_VERSION:
            raise ValueError(
                f"label space was built with unicode spec {d['spec_version']} "
                f"but this code is {SPEC_VERSION}; checkpoints are not portable "
                f"across spec versions")
        return cls(
            chars=d["chars"], shapes=d["shapes"], tones=d["tones"],
            langs=d["langs"],
            legal={l: {b: [tuple(p) for p in ps] for b, ps in dd.items()}
                   for l, dd in d["legal"].items()},
            spec_version=d["spec_version"],
        )


# ---------------------------------------------------------------------------
# construction from a corpus
# ---------------------------------------------------------------------------

class LabelSpaceBuilder:
    """Streams (text, lang) pairs and accumulates the observed label space."""

    def __init__(self, min_char_count: int = 20, min_combo_count: int = 3):
        self.min_char_count = min_char_count
        self.min_combo_count = min_combo_count
        self.char_counts: Counter = Counter()
        self.shape_counts: Counter = Counter()
        self.tone_counts: Counter = Counter()
        self.combo_counts: Dict[str, Counter] = defaultdict(Counter)
        self.langs_seen: set = set()

    def add(self, text: str, lang: str) -> None:
        if lang not in SPEC:
            return
        self.langs_seen.add(lang)
        for g in graphemes(text):
            b, s, t = factorize_char(g, lang)
            if b == g and not s and not t and len(g) > 1:
                continue          # atomic (foreign script): no class for it
            if not in_inventory(g, lang):
                continue          # another language's alphabet: no class
            self.char_counts[g] += 1
            self.char_counts[b] += 1          # the stripped form is also input
            self.shape_counts[_shape_key(s)] += 1
            self.tone_counts[t] += 1
            self.combo_counts[lang][(b, _shape_key(s), t)] += 1

    def build(self) -> LabelSpace:
        chars = list(SPECIAL_TOKENS)
        chars += [c for c, n in self.char_counts.most_common()
                  if n >= self.min_char_count and c not in SPECIAL_TOKENS]

        shapes = ["NONE"] + sorted(
            k for k, n in self.shape_counts.items()
            if k != "NONE" and n >= self.min_combo_count)
        tones = [""] + sorted(
            k for k, n in self.tone_counts.items()
            if k != "" and n >= self.min_combo_count)

        s2i = {s: i for i, s in enumerate(shapes)}
        t2i = {t: i for i, t in enumerate(tones)}

        legal: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}
        for lang, counter in self.combo_counts.items():
            per_base: Dict[str, set] = defaultdict(set)
            for (b, sk, t), n in counter.items():
                if n < self.min_combo_count:
                    continue
                if sk not in s2i or t not in t2i:
                    continue
                per_base[b].add((s2i[sk], t2i[t]))
            # identity is always legal -- a user may legitimately want the bare
            # letter even if the corpus never showed it bare
            for b in list(per_base):
                per_base[b].add((0, 0))
            legal[lang] = {b: sorted(v) for b, v in per_base.items()}

        langs = [l for l in LANGS if l in self.langs_seen] or list(LANGS)
        return LabelSpace(chars=chars, shapes=shapes, tones=tones,
                          langs=langs, legal=legal)


def build_from_spec_only() -> LabelSpace:
    """A corpus-free label space, useful for tests and for bootstrapping before
    the first corpus scan. Enumerates every (base, shape, tone) implied by the
    declarative language spec over ASCII letters."""
    b = LabelSpaceBuilder(min_char_count=1, min_combo_count=1)
    import string
    from .unicode_ops import (ALL_TONE_MARKS, MARK_NAMES, SPECIAL_LETTERS,
                              compose_char)
    for lang, spec in SPEC.items():
        for base in string.ascii_letters + " .,;:!?'\"-()0123456789\n":
            b.add(base, lang)
            for m in spec.all_marks:
                c = compose_char(base, (), m) if m in spec.tone_marks else \
                    compose_char(base, (MARK_NAMES[m],), "")
                if c and c != base:
                    b.add(c, lang)
                for t in spec.tone_marks:
                    if m in spec.tone_marks:
                        continue
                    c2 = compose_char(base, (MARK_NAMES[m],), t)
                    if c2 and c2 != base:
                        b.add(c2, lang)
        for letter, (bs, tag) in SPECIAL_LETTERS.items():
            b.add(letter, lang)
    return b.build()
