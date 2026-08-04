"""
Inference.

Everything here exists because of something that breaks in production rather
than on a benchmark:

  sliding windows      benchmark sentences are 120 characters; real inputs are
                       documents. Windows overlap and only the centre of each
                       window is trusted, so every character is predicted with
                       context on both sides.
  legality masking     the model may only emit (shape, tone) pairs observed for
                       that base character in that language.
  protected spans      URLs, emails, handles, code and identifiers are passed
                       through untouched. An accent inside a URL is never right.
  abstention           per-character confidence with a threshold. Below it the
                       character is left as the user wrote it. A wrong tone mark
                       changes meaning; a missing one is merely incomplete, and
                       for most business copy that trade is the correct one.
  lexicon snapping     if the predicted word is not a word, and the stripped
                       form has known diacritized variants, re-score against
                       them. Cheap, and it removes the non-word outputs that
                       are the most visible class of error to a reader.
  respect_existing     marks already present in the input are treated as the
                       user's intent and preserved by default.

The structural guarantee -- strip(output) == strip(input) -- is asserted at the
end of every call, not assumed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .labels import LabelSpace, UNK_ID
from .unicode_ops import (base_char, graphemes, in_inventory, nfc,
                          normalize_lang, strip_diacritics)

# Upstream keeps this in diactag/data.py alongside the training pipeline. Only
# the regex is needed at inference time, so it is inlined here rather than
# vendoring the whole data module.
#
# Spans that must never be touched: they are not natural language and a
# "restored" accent inside them is always a bug.
PROTECTED_RE = re.compile(
    r"""(https?://\S+|www\.\S+                 # urls
        |[\w.+-]+@[\w-]+\.[\w.]+               # emails
        |[@#][\w_]+                            # handles / hashtags
        |`[^`]*`|<[^>\s]{1,40}>                # inline code, tags
        |\b[A-Z]{2,}(?:_[A-Z0-9]+)+\b          # CONSTANT_NAMES
        |\b\w+\.(?:com|org|net|io|ng|vn)\b     # bare domains
        )""",
    re.VERBOSE,
)


# ---------------------------------------------------------------------------
# window planning (pure, unit-tested)
# ---------------------------------------------------------------------------

def plan_windows(n: int, window: int, stride: int
                 ) -> List[Tuple[int, int, int, int]]:
    """Cover [0, n) with overlapping windows.

    Returns (start, end, accept_start, accept_end) where the accept range is
    the part of the window whose predictions we keep. The accepted ranges
    partition [0, n) exactly: every position is predicted once, from the
    window in which it sits furthest from an edge.
    """
    if n <= window:
        return [(0, n, 0, n)]
    starts = list(range(0, max(1, n - window + 1), stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    out: List[Tuple[int, int, int, int]] = []
    prev_end = 0
    for i, s in enumerate(starts):
        e = min(n, s + window)
        if i == len(starts) - 1:
            acc_end = n
        else:
            nxt = starts[i + 1]
            acc_end = min(e, (nxt + e) // 2)   # hand over at the midpoint
        acc_start = max(prev_end, s)
        if acc_end <= acc_start:
            continue
        out.append((s, e, acc_start, acc_end))
        prev_end = acc_end
    return out


def protected_grapheme_flags(text_graphemes: Sequence[str]) -> List[bool]:
    text = "".join(text_graphemes)
    flags = [False] * len(text_graphemes)
    spans = [(m.start(), m.end()) for m in PROTECTED_RE.finditer(text)]
    if not spans:
        return flags
    ci = 0
    si = 0
    for gi, g in enumerate(text_graphemes):
        while si < len(spans) and spans[si][1] <= ci:
            si += 1
        if si < len(spans) and spans[si][0] <= ci < spans[si][1]:
            flags[gi] = True
        ci += len(g)
    return flags


WORD_RE = re.compile(r"\w+", re.UNICODE)


# ---------------------------------------------------------------------------

@dataclass
class InferConfig:
    window: int = 384
    stride: int = 256
    batch_size: int = 32
    min_confidence: float = 0.0        # 0 disables abstention
    abstain_policy: str = "input"      # "input" keeps user's marks, "bare" strips
    respect_existing: bool = True      # never remove a mark the user typed
    use_legality: bool = True
    lexicon_mode: str = "off"          # off | rerank
    # How far below the model's own score an attested word may sit and still be
    # accepted. The model's output is the per-character argmax, so every
    # candidate scores LOWER by construction -- the margin is a tolerance, not a
    # bar to clear. Larger = more willing to trust the lexicon over the model.
    lexicon_margin: float = 2.0
    device: str = "cpu"
    temperature: float = 1.0           # set by calibrate.py


@dataclass
class CharResult:
    char: str
    confidence: float
    abstained: bool
    protected: bool


class Diacritizer:
    def __init__(self, model, ls: LabelSpace, cfg: Optional[InferConfig] = None,
                 lexicon=None):
        import numpy as np
        import torch
        self.torch = torch
        self.np = np
        self.model = model.eval()
        self.ls = ls
        self.cfg = cfg or InferConfig()
        self.lexicon = lexicon
        self.model.to(self.cfg.device)
        legal = ls.legal_mask_array()                       # [L, C, S, T]
        self.legal = torch.from_numpy(legal).to(self.cfg.device)
        self._id_lookup = ls.char2id

    # -- language identification ----------------------------------------
    @property
    def torch_no_grad(self):
        return self.torch.no_grad()

    def detect_language(self, text: str) -> Tuple[str, float]:
        torch = self.torch
        gs = graphemes(text)[: self.cfg.window]
        ids = torch.tensor([[self._id_lookup.get(g, UNK_ID) for g in gs]],
                           device=self.cfg.device)
        lang = torch.zeros(1, dtype=torch.long, device=self.cfg.device)
        known = torch.zeros(1, dtype=torch.long, device=self.cfg.device)
        with torch.no_grad():
            out = self.model(ids, lang, known, need_mlm=False)
        p = out["lid"].softmax(-1)[0]
        i = int(p.argmax())
        return self.ls.langs[i], float(p[i])

    # -- main entry point -------------------------------------------------
    def restore(self, text: str, lang: Optional[str] = None,
                return_details: bool = False):
        torch = self.torch
        cfg = self.cfg
        text = nfc(text)
        gs = graphemes(text)
        if not gs:
            return ("", []) if return_details else ""

        lang = normalize_lang(lang)
        if lang is None:
            lang, _ = self.detect_language(text)
        li = self.ls.lang2id[lang]

        ids_all = [self._id_lookup.get(g, UNK_ID) for g in gs]
        n = len(gs)
        shape_lp = torch.full((n, self.ls.n_shapes), -1e30)
        tone_lp = torch.full((n, self.ls.n_tones), -1e30)

        plans = plan_windows(n, cfg.window, cfg.stride)
        for b0 in range(0, len(plans), cfg.batch_size):
            chunk = plans[b0:b0 + cfg.batch_size]
            L = max(e - s for s, e, _, _ in chunk)
            ids = torch.full((len(chunk), L), 0, dtype=torch.long)
            attn = torch.zeros((len(chunk), L), dtype=torch.bool)
            for i, (s, e, _, _) in enumerate(chunk):
                ids[i, :e - s] = torch.tensor(ids_all[s:e])
                attn[i, :e - s] = True
            ids = ids.to(cfg.device)
            attn = attn.to(cfg.device)
            lt = torch.full((len(chunk),), li, dtype=torch.long, device=cfg.device)
            kt = torch.ones_like(lt)
            with torch.no_grad():
                out = self.model(ids, lt, kt, attn, need_mlm=False)
            sl = (out["shape"] / cfg.temperature).log_softmax(-1).float().cpu()
            tl = (out["tone"] / cfg.temperature).log_softmax(-1).float().cpu()
            for i, (s, e, a0, a1) in enumerate(chunk):
                shape_lp[a0:a1] = sl[i, a0 - s:a1 - s]
                tone_lp[a0:a1] = tl[i, a0 - s:a1 - s]

        # joint scores with legality masking
        joint = shape_lp[:, :, None] + tone_lp[:, None, :]      # [n, S, T]
        if cfg.use_legality:
            cid = torch.tensor(ids_all)
            mask = self.legal[li].cpu()[cid]                    # [n, S, T]
            joint = joint.masked_fill(~mask, -1e30)
        flat = joint.reshape(n, -1)
        probs = flat.softmax(-1)
        best = flat.argmax(-1)
        conf = probs.gather(1, best[:, None]).squeeze(1)
        s_idx = (best // self.ls.n_tones).tolist()
        t_idx = (best % self.ls.n_tones).tolist()
        conf_l = conf.tolist()

        protected = protected_grapheme_flags(gs)
        # A character that belongs to another language's alphabet is a foreign
        # word, not something to re-accent. Without this, Turkish input
        # containing "Việt" comes back mangled toward Turkish orthography.
        for i, g in enumerate(gs):
            if g != base_char(g) and not in_inventory(g, lang):
                protected[i] = True
        details: List[CharResult] = []
        out_chars: List[str] = []
        for i, g in enumerate(gs):
            b = base_char(g)
            had_mark = (g != b)
            if protected[i]:
                out_chars.append(g)
                details.append(CharResult(g, 1.0, False, True))
                continue
            c = self.ls.decode_char(b, s_idx[i], t_idx[i])
            abstain = cfg.min_confidence > 0 and conf_l[i] < cfg.min_confidence
            if abstain:
                c = g if cfg.abstain_policy == "input" else b
            if cfg.respect_existing and had_mark and c == b:
                c = g          # never silently delete a mark the user typed
            out_chars.append(c)
            details.append(CharResult(c, conf_l[i], abstain, False))

        if cfg.lexicon_mode == "rerank" and self.lexicon is not None:
            out_chars = self._lexicon_rerank(gs, out_chars, joint, lang, details)

        result = "".join(out_chars)
        # structural guarantee, verified rather than assumed
        if strip_diacritics(result) != strip_diacritics(text):
            raise RuntimeError("strip invariant violated -- this is a bug")
        return (result, details) if return_details else result

    # -- lexicon ----------------------------------------------------------
    def _lexicon_rerank(self, gs, out_chars, joint, lang, details):
        text = "".join(out_chars)
        # grapheme index of each codepoint start
        starts: List[int] = []
        pos = 0
        for gi, c in enumerate(out_chars):
            starts.append(pos)
            pos += len(c)
        for m in WORD_RE.finditer(text):
            gi0 = next((i for i, p in enumerate(starts) if p >= m.start()), None)
            if gi0 is None:
                continue
            gi1 = gi0
            while gi1 < len(out_chars) and starts[gi1] < m.end():
                gi1 += 1
            word = "".join(out_chars[gi0:gi1])
            cands = self.lexicon.variants(lang, word)
            if not cands or word in cands:
                continue
            best, best_score = None, -1e30
            cur = self._score_word(joint, gi0, gi1, word, lang)
            bare = strip_diacritics(word)
            for cand, _freq in cands.items():
                # The candidate must have EXACTLY the same base characters,
                # case included. The lexicon is keyed on the lowercased strip,
                # so `OGUN` retrieves `Ògùn` as well as `ÒGÙN` -- substituting
                # the first would silently change the text's capitalisation and
                # break the one guarantee this architecture exists to provide.
                if strip_diacritics(cand) != bare:
                    continue
                if len(graphemes(cand)) != gi1 - gi0:
                    continue
                sc = self._score_word(joint, gi0, gi1, cand, lang)
                if sc > best_score:
                    best, best_score = cand, sc
            # Only reached when the model emitted a NON-word whose stripped form
            # has attested spellings. Accept the best attested one unless it is
            # far less likely than what the model chose.
            if best is not None and best_score > cur - self.cfg.lexicon_margin:
                for k, g in enumerate(graphemes(best)):
                    out_chars[gi0 + k] = g
                    details[gi0 + k].char = g
        return out_chars

    def _score_word(self, joint, gi0: int, gi1: int, word: str, lang: str) -> float:
        total = 0.0
        for k, g in enumerate(graphemes(word)):
            from .unicode_ops import factorize_char
            b, s, t = factorize_char(g, lang)
            si = self.ls.shape2id.get("+".join(s) if s else "NONE")
            ti = self.ls.tone2id.get(t)
            if si is None or ti is None:
                return -1e30
            total += float(joint[gi0 + k, si, ti])
        return total

    # -- batch ------------------------------------------------------------
    def restore_batch(self, texts: Sequence[str],
                      langs: Optional[Sequence[Optional[str]]] = None
                      ) -> List[str]:
        langs = langs or [None] * len(texts)
        return [self.restore(t, l) for t, l in zip(texts, langs)]
