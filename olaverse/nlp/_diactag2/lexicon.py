"""
Per-language word lexicon.

Maps stripped word form -> {diacritized variants: frequency}. Two uses:

1. Inference-time reranking. If the model emits a string that is not a word in
   the language but the stripped form has known variants, re-score the
   candidates under the model's own per-character distribution and take the
   best. This never invents a word: it only ever chooses among forms attested
   in the corpus, and only when the model's own preferred output was not one.

2. Measurement. "Fraction of output words that are not attested" is a metric a
   non-technical reviewer can act on, unlike DER.

Deliberately conservative: single-variant entries below a frequency floor are
dropped (they are usually corpus noise), and capitalised words are only
reranked if the capitalised form itself is attested, so surnames and place
names are not nativised.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

from .unicode_ops import base_char, factorize_char, graphemes, nfc, strip_diacritics

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


class Lexicon:
    def __init__(self, table: Optional[Dict[str, Dict[str, Dict[str, int]]]] = None,
                 density_gated: bool = False):
        # table[lang][stripped_lower] = {variant: count}
        self.table = table or {}
        # Whether spurious variants were already excluded at build time by
        # restricting to densely-marked sentences. If so, a bare variant in the
        # table is a real bare word and must NOT be filtered further.
        self.density_gated = density_gated

    # -- build -----------------------------------------------------------
    @classmethod
    def build(cls, rows: Iterable, min_count: int = 3, max_variants: int = 8,
              density_floors: Optional[Dict[str, float]] = None) -> "Lexicon":
        """Build from `rows` of {text, lang}.

        `density_floors` is the important argument. Without it the lexicon is
        built from the whole corpus, and in an under-marked corpus that makes
        the ambiguity statistic meaningless: every word acquires a bare variant,
        so `ile`/`ilé` looks like a minimal pair when it is one word typed two
        ways.

        Restricting to densely-marked sentences fixes this at the source. In
        text where the writer *was* marking diacritics, a bare word is a real
        bare word -- so Italian `studio` vs `studiò` survives (both attested in
        careful text) while Yoruba `ile` vs `ilé` collapses if the bare form
        only ever appears in sloppy text.
        """
        from .unicode_ops import diacritic_density
        floors = density_floors or {}
        acc: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        for r in rows:
            text, lang = nfc(r["text"]), r["lang"]
            f = floors.get(lang)
            if f is not None and diacritic_density(text, lang) < f:
                continue
            for m in WORD_RE.finditer(text):
                w = m.group(0)
                acc[lang][strip_diacritics(w).lower()][w] += 1
        table: Dict[str, Dict[str, Dict[str, int]]] = {}
        for lang, d in acc.items():
            out: Dict[str, Dict[str, int]] = {}
            for key, counter in d.items():
                kept = {w: c for w, c in counter.most_common(max_variants)
                        if c >= min_count}
                if kept:
                    out[key] = kept
            table[lang] = out
        return cls(table, density_gated=bool(floors))

    # -- query -----------------------------------------------------------
    def variants(self, lang: str, word: str) -> Dict[str, int]:
        d = self.table.get(lang)
        if not d:
            return {}
        key = strip_diacritics(word).lower()
        cands = d.get(key, {})
        if not cands:
            return {}
        if word[:1].isupper():
            # only rerank capitalised tokens among capitalised attestations
            cands = {w: c for w, c in cands.items() if w[:1].isupper()}
        return cands

    def is_attested(self, lang: str, word: str) -> bool:
        return word in self.variants(lang, word)

    # -- ambiguity --------------------------------------------------------
    @staticmethod
    def _dominates(a: str, b: str, lang: str) -> bool:
        """True if `b` is `a` with some marks removed -- i.e. b is a
        partially-stripped spelling of a, not a different word."""
        ga, gb = graphemes(a), graphemes(b)
        if len(ga) != len(gb):
            return False
        for x, y in zip(ga, gb):
            if x == y:
                continue
            bx, sx, tx = factorize_char(x, lang)
            by, sy, ty = factorize_char(y, lang)
            if bx != by:
                return False
            # every mark on b must also be on a; b may drop marks, not add or
            # change them
            if not set(sy).issubset(sx):
                return False
            if ty and ty != tx:
                return False
        return True

    def true_variants(self, lang: str, key: str) -> Dict[str, int]:  # noqa: C901
        """Variants that are genuinely different words, not the same word typed
        with fewer marks.

        This distinction is the whole difference between a useful ambiguity
        statistic and a corpus-quality artefact. A corpus containing
        under-marked text lists {ogun, ogún, ògùn} for one stripped form, and
        counting that as three-way ambiguity says nothing about Yoruba -- two
        of those are the same word missing its tone. Real ambiguity is
        `ogún` (twenty) vs `ògùn` (medicine): same skeleton, incompatible marks.
        """
        cands = self.table.get(lang, {}).get(key, {})
        if len(cands) < 2:
            return dict(cands)
        if self.density_gated:
            # Already handled at the source, and applying the heuristic now
            # would be wrong: Italian `studio` (noun) is a mark-stripped form of
            # `studiò` (verb) by shape alone, but both are real words, and in
            # carefully-marked text both are attested as themselves.
            return dict(cands)
        # drop any variant that is a mark-stripped version of a richer one
        keep = {}
        for w, c in cands.items():
            if any(other != w and self._dominates(other, w, lang)
                   for other in cands):
                continue
            keep[w] = c
        return keep

    def ambiguity_stats(self, lang: str) -> Dict[str, float]:
        """Type- and token-level ambiguity.

        The token figure is the one that bounds achievable accuracy. Type-level
        ambiguity counts each distinct word once, so a rare ambiguous form
        weighs as much as `ni` or `e` -- but the model is scored on running
        text, where a handful of very frequent ambiguous function words matter
        far more than a long tail of rare ones. `token_amb_frac` is the share of
        *occurrences* that land on an ambiguous form, and no context-free system
        can beat it; a context-using one does better only where context
        actually disambiguates.

        Note `spurious_frac` is meaningful only for an ungated lexicon. When the
        lexicon was density-gated the spurious variants were removed at build
        time, so this is identically zero by construction, not by measurement.
        """
        d = self.table.get(lang, {})
        raw = sum(1 for v in d.values() if len(v) > 1)
        true = 0
        amb_tokens = 0
        all_tokens = 0
        for k, v in d.items():
            tot = sum(v.values())
            all_tokens += tot
            if len(v) > 1 and len(self.true_variants(lang, k)) > 1:
                true += 1
                amb_tokens += tot
        n = max(1, len(d))
        return {"forms": len(d), "raw_ambiguous": raw, "true_ambiguous": true,
                "raw_frac": round(raw / n, 4), "true_frac": round(true / n, 4),
                "token_amb_frac": round(amb_tokens / max(1, all_tokens), 4),
                "spurious_frac": (round((raw - true) / max(1, raw), 4)
                                  if not self.density_gated else None)}

    def oov_rate(self, text: str, lang: str) -> float:
        words = WORD_RE.findall(nfc(text))
        if not words:
            return 0.0
        known = self.table.get(lang, {})
        miss = 0
        for w in words:
            v = known.get(strip_diacritics(w).lower())
            if v is None:
                continue          # unknown stripped form: cannot judge
            if w not in v:
                miss += 1
        return miss / len(words)

    # -- io ---------------------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"density_gated": self.density_gated,
                       "table": self.table}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict) and "table" in d:
            return cls(d["table"], density_gated=d.get("density_gated", False))
        return cls(d)                    # legacy format

    def stats(self, true_ambiguity: bool = True) -> Dict[str, Dict[str, float]]:
        if not true_ambiguity:
            out = {}
            for lang, d in self.table.items():
                amb = sum(1 for v in d.values() if len(v) > 1)
                out[lang] = {"forms": len(d), "ambiguous": amb,
                             "ambiguous_frac": round(amb / max(1, len(d)), 4)}
            return out
        return {lang: self.ambiguity_stats(lang) for lang in self.table}
