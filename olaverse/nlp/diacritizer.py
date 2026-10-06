"""
DiacNet / DiacTag — Runtime Diacritization Engine
=================================================
Provides a unified Diacritizer class to load and use any of the available
diacritization models.

Supported Methods for Yoruba ('yo'):
  - "viterbi": Fast statistical n-gram Viterbi decoder (default)
  - "knn": Character k-NN backoff (dot-below only)
  - "bilstm": High-accuracy character-level BiLSTM
  - "transformer": High-accuracy XLM-RoBERTa Transformer

Supported Methods for Igbo ('ig'):
  - "knn": Character k-NN backoff (default)

Multilingual (10 languages: yo, ig, ha, vi, pl, tr, pt, es, fr, it):
  - "diacnet": ByT5 seq2seq — diacnet-1.0, diacnet-1.1
  - "diactag": per-character tagger — diactag-1.0

Multilingual with Arabic (11 languages: the ten above plus ar):
  - "diactag": per-character tagger — diactag-2.0
  - "diacnet2": ByT5 text-to-text — diacnet-2.0, diacnet-mini-2.0

The two multilingual families differ in kind, not degree. A seq2seq model
generates the output text, so it *can* drop a word or rewrite a clause; on
Hausa only 94.7% of diacnet-1.1's outputs still stripped back to their input.
The tagger classifies each character into a diacritic transformation and copies
the base character, so ``strip(output) == strip(input)`` holds by construction
and is asserted on every call. The price is that a tagger cannot insert or
delete characters, so it cannot fix a typo.

For diacnet-1.x that meant preferring diactag unless you needed typo repair, or
were on Vietnamese or Portuguese. The diacnet-2.x models change the trade: with
*output alignment* (on by default) the model's text is aligned back onto yours,
keeping your letters and taking only its marks, so they no longer change the
text either. Per the model cards, diacnet-2.0 then scores a lower error rate than
diactag-2.0 on 8 of the 10 Latin-script languages, at 582M parameters against
37.9M; diactag-2.0 stays the CPU-native choice, and the better one on Igbo,
Vietnamese and Modern Standard Arabic. diacnet-1.x output is left exactly as it
was unless you ask for ``aligned=True``.

diactag-1.0 and diactag-2.0 are decoded by different vendored code, chosen from
the ``spec_version`` recorded in each checkpoint's ``labels.json``.
"""

import os
import json
import unicodedata
import re
from typing import List, Tuple, Union
from olaverse.utils.downloader import get_model_path

_YORUBA_MODEL_CACHE = {}
_YORUBA_DB_MODEL_CACHE = {}
_IGBO_MODEL_CACHE = {}
_NEURAL_CACHE = {}

def remove_tones(text):
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(
        c for c in decomposed 
        if unicodedata.category(c) != 'Mn' or ord(c) == 0x0323
    )
    return unicodedata.normalize('NFC', filtered)

def strip_all_diacritics(text):
    decomposed = unicodedata.normalize('NFD', text)
    filtered = "".join(
        c for c in decomposed 
        if unicodedata.category(c) != 'Mn'
    )
    return unicodedata.normalize('NFC', filtered)

def _load_diacritizer_model(path, is_custom=False):
    if not os.path.exists(path):
        if is_custom:
            raise FileNotFoundError(f"Diacritizer model file not found at: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── Viterbi Decoder ───
def viterbi_decode(text, model):
    candidates_map = model.get("candidates", {})
    transitions = model.get("transitions", {})
    unigrams = model.get("unigrams", {})
    
    tokens = re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip() and re.match(r'^[\w\u0300-\u036f]+$', t)]
    
    if not word_indices:
        return text
        
    dp = []
    first_word_idx = word_indices[0]
    first_token = tokens[first_word_idx]
    first_token_lower = first_token.lower()
    
    candidates = candidates_map.get(first_token_lower, [first_token_lower])
    first_dp = {}
    for cand in candidates:
        if first_token.isupper():
            cand_formatted = cand.upper()
        elif first_token[0].isupper():
            cand_formatted = cand.capitalize()
        else:
            cand_formatted = cand
            
        unigram_prob = unigrams.get(cand, -12.0)
        first_dp[cand_formatted] = (unigram_prob, None)
    dp.append(first_dp)
    
    for step_idx in range(1, len(word_indices)):
        prev_word_idx = word_indices[step_idx - 1]
        curr_word_idx = word_indices[step_idx]
        curr_token = tokens[curr_word_idx]
        curr_token_lower = curr_token.lower()
        
        curr_candidates = candidates_map.get(curr_token_lower, [curr_token_lower])
        curr_dp = {}
        for cand in curr_candidates:
            if curr_token.isupper():
                cand_formatted = cand.upper()
            elif curr_token[0].isupper():
                cand_formatted = cand.capitalize()
            else:
                cand_formatted = cand
                
            best_prob = -float('inf')
            best_prev = None
            
            for prev_cand, (prev_prob, _) in dp[-1].items():
                prev_cand_lower = prev_cand.lower()
                transition_key = f"{prev_cand_lower} {cand}"
                trans_prob = transitions.get(transition_key, unigrams.get(cand, -12.0) - 5.0)
                
                total_prob = prev_prob + trans_prob
                if total_prob > best_prob:
                    best_prob = total_prob
                    best_prev = prev_cand
                    
            curr_dp[cand_formatted] = (best_prob, best_prev)
        dp.append(curr_dp)
        
    best_final_prob = -float('inf')
    best_final_cand = None
    for cand, (prob, _) in dp[-1].items():
        if prob > best_final_prob:
            best_final_prob = prob
            best_final_cand = cand
            
    best_path = [best_final_cand]
    for step_idx in range(len(word_indices) - 1, 0, -1):
        prev_cand = dp[step_idx][best_path[-1]][1]
        best_path.append(prev_cand)
        
    best_path.reverse()
    
    output_tokens = list(tokens)
    for idx, word_idx in enumerate(word_indices):
        output_tokens[word_idx] = best_path[idx]
        
    return "".join(output_tokens)

# ─── KNN Decoder ───
def knn_decode(text, model, target_chars):
    db_5 = model.get("db_5", {})
    db_3 = model.get("db_3", {})
    db_1 = model.get("db_1", {})
    
    tokens = re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)
    word_indices = [i for i, t in enumerate(tokens) if t.strip() and re.match(r'^[\w\u0300-\u036f]+$', t)]
    
    if not word_indices:
        return text
        
    def get_context(word, i, W=2):
        left = word[max(0, i-W) : i]
        left = "_" * (W - len(left)) + left
        right = word[i+1 : min(len(word), i+1+W)]
        right = right + "_" * (W - len(right))
        return left + word[i] + right
        
    def predict_backoff(ctx_query):
        if ctx_query in db_5: return db_5[ctx_query]
        ctx_3 = ctx_query[1:4]
        if ctx_3 in db_3: return db_3[ctx_3]
        target = ctx_query[2]
        return db_1.get(target, target)
        
    output_tokens = list(tokens)
    for word_idx in word_indices:
        token = tokens[word_idx]
        token_lower = token.lower()
        
        pred_chars = list(token_lower)
        for i in range(len(token_lower)):
            if token_lower[i] in target_chars:
                ctx_query = get_context(token_lower, i, 2)
                pred_chars[i] = predict_backoff(ctx_query)
                
        pred_word = "".join(pred_chars)
        if token.isupper():
            pred_word_formatted = pred_word.upper()
        elif token[0].isupper():
            pred_word_formatted = pred_word.capitalize()
        else:
            pred_word_formatted = pred_word
            
        output_tokens[word_idx] = pred_word_formatted
        
    return "".join(output_tokens)

# ─── Neural Decoders ───
class BiLSTMDecoder:
    def __init__(self, pt_path, vocab_path):
        import torch
        import torch.nn as nn
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        
        class DiacNetCharModel(nn.Module):
            def __init__(self, vocab_size, emb_dim=64, hidden_dim=256):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True, num_layers=2)
                self.classifier = nn.Linear(hidden_dim, 6)
            def forward(self, char_seqs, lengths):
                emb = self.embedding(char_seqs)
                packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
                out, _ = self.lstm(packed)
                out, _ = pad_packed_sequence(out, batch_first=True)
                return self.classifier(out)

        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
        self.char_vocab = v_data["char_vocab"]
        self.word_candidates = v_data["word_candidates"]
        
        state = torch.load(pt_path, map_location=self.device)
        self.model = DiacNetCharModel(len(self.char_vocab)).to(self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

    def decode(self, text):
        import torch
        text_nfd = unicodedata.normalize('NFD', text)
        base_chars = [c for c in text_nfd if unicodedata.category(c) != 'Mn']
        if not base_chars: return text
        
        char_ids = [self.char_vocab.get(c, 1) for c in base_chars]
        tensor_in = torch.tensor([char_ids], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(char_ids)], dtype=torch.long)
        
        with torch.no_grad():
            logits = self.model(tensor_in, lengths)
            preds = logits.argmax(dim=-1).squeeze(0).tolist()
            
        parts = []
        for c, l in zip(base_chars, preds):
            if l == 0 or not c.isalpha(): parts.append(c)
            elif l == 1: parts.append(c + '\u0323')
            elif l == 2: parts.append(c + '\u0301')
            elif l == 3: parts.append(c + '\u0300')
            elif l == 4: parts.append(c + '\u0323\u0301')
            elif l == 5: parts.append(c + '\u0323\u0300')
        pred_sentence = unicodedata.normalize('NFC', "".join(parts))
        
        plain_words = re.findall(r'\S+', strip_all_diacritics(text))
        pred_words = re.findall(r'\S+', pred_sentence)
        tokens = re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)
        word_indices = [i for i, t in enumerate(tokens) if t.strip() and re.match(r'^[\w\u0300-\u036f]+$', t)]
        
        for w_idx, pw, pw_pred in zip(word_indices, plain_words, pred_words):
            pw_l = pw.lower()
            cands = self.word_candidates.get(pw_l, [])
            if not cands or pw_pred.lower() in cands:
                corrected = pw_pred
            else:
                corrected = cands[0]
                if pw_pred and pw_pred[0].isupper():
                    corrected = corrected.capitalize()
            tokens[w_idx] = corrected
            
        return "".join(tokens)

_DIACNET_LANG_TAGS = {
    "yo": "<yor>", "vi": "<vie>", "ig": "<ibo>", "ha": "<hau>", "pl": "<pol>",
    "tr": "<tur>", "pt": "<por>", "es": "<spa>", "fr": "<fra>", "it": "<ita>",
}

# diacnet-1.0 was trained on sentence-length input (median 58 bytes), so long
# text is split on sentence boundaries and restored piece by piece. The lookbehind
# keeps the terminator attached to the sentence it belongs to.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list:
    """Default sentence splitter used by :class:`DiacNetDecoder`."""
    return [p for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]


class DiacNetDecoder:
    """
    diacnet-1.0 diacritic restoration (byte-level seq2seq) — one joint model, 10 languages.

    Trained on sentence-length input (median 58 bytes), so multi-sentence text is
    split on sentence boundaries and restored a sentence at a time, then rejoined.
    Pass ``split_sentences=False`` to send the whole string in one pass, or supply
    your own callable to control the boundaries::

        DiacNetDecoder(splitter=my_splitter)          # custom segmentation
        decoder.decode(text, split_sentences=False)   # one pass, no splitting

    Being seq2seq rather than a per-character tagger, it can rewrite text instead
    of only adding marks. Very short fragments are below the trained input length
    and degenerate — repetition loops ("el nino" -> "el niño\\nel niño\\n..."),
    changed inflections ("nino" -> "niños"), or another language's diacritics
    ("cafe" with lang="fr" -> "cafẹ́"). Restores diacritics only; apostrophes and
    other punctuation are not inserted.
    """

    #: Input longer than this many tokens is truncated.
    MAX_INPUT_TOKENS = 256

    def __init__(self, model_name="olaverse/diacnet-1.0", splitter=None):
        """
        Args:
            model_name: Hugging Face model id.
            splitter: Callable taking a string and returning a list of segments.
                      Defaults to splitting on sentence-ending punctuation.
        """
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.splitter = splitter or split_sentences

    def decode(self, text: str, lang: str = "yo", max_new_tokens: int = 256,
               split_sentences: bool = True, splitter=None,
               aligned: bool = False) -> str:
        """
        Restore diacritics.

        Args:
            text: Input text. Multi-sentence input is segmented by default.
            lang: One of the 10 supported language codes.
            max_new_tokens: Generation budget per segment.
            split_sentences: Set False to run the whole string through in one
                             pass, bypassing segmentation.
            splitter: Per-call override for the segmentation callable. Passed
                      here rather than held on the instance so a shared cached
                      decoder isn't mutated by one caller's choice.
            aligned: Keep each segment's own letters and take only the model's
                     marks (see :func:`olaverse.nlp.diacnet_utils.align`), so
                     the output strips back to the input. Off by default, which
                     leaves diacnet-1.x output exactly as it has always been.

        Returns:
            str: the restored text.
        """
        tag = _DIACNET_LANG_TAGS.get(lang.lower())
        if tag is None:
            raise ValueError(
                f"Unsupported language '{lang}' for diacnet-1.0. "
                f"Supported: {sorted(_DIACNET_LANG_TAGS)}"
            )

        text = (text or "").strip()
        if not text:
            return ""

        if not split_sentences:
            return self._decode_one(tag, text, max_new_tokens, aligned)

        segments = (splitter or self.splitter)(text)
        if len(segments) <= 1:
            return self._decode_one(tag, text, max_new_tokens, aligned)
        return " ".join(self._decode_one(tag, s, max_new_tokens, aligned) for s in segments)

    def _decode_one(self, tag: str, text: str, max_new_tokens: int,
                    aligned: bool = False) -> str:
        import torch

        inputs = self.tokenizer(
            f"{tag} {text.strip()}",
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_INPUT_TOKENS,
        )
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        out = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if aligned:
            from olaverse.nlp.diacnet_utils import align
            out = align(text.strip(), out.strip())
        return out


class DiacNet2Decoder:
    """
    diacnet-2.0 / diacnet-mini-2.0 diacritic restoration (byte-level seq2seq) —
    11 languages including Arabic, from one model.

    A ByT5 text-to-text model. The prompt is ``"<tag> [g: word=meaning] text"``;
    the language tag is optional (``<auto>``) and so is the gloss block. Unlike
    diacnet-1.x it was trained on pieces of up to ~300 characters, so text is
    split into chunks of at most that size on spaces (not on sentence ends),
    decoded greedily, and the pieces rejoined with single spaces.

    By default (``aligned=True``) the output keeps the input's own letters and
    takes only the model's marks, so ``strip_marks(output) == strip_marks(input)``
    by construction. ``aligned=False`` returns the raw generation, which can also
    repair typos but may change a letter in a few percent of sentences.

    Batching: chunks from every text are pooled, sorted by length and padded to
    the longest in each batch, so many short texts cost far less than one call
    each. On a CUDA device the weights load in bfloat16; elsewhere in float32.
    The reference outputs were produced by batched bf16 GPU runs, so a CPU
    float32 run can differ by an occasional mark.
    """

    #: Longest piece sent to the model in one call, in characters.
    CHUNK_CHARS = 300

    def __init__(self, model_name: str = "olaverse/diacnet-2.0",
                 device: str = None, batch_size: int = 16):
        """
        Args:
            model_name: Hugging Face model id.
            device: ``"cuda"``, ``"cpu"``, ``"mps"`` ... ``None`` or ``"auto"``
                    picks CUDA when available, else CPU.
            batch_size: Chunks decoded together.
        """
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, int(batch_size))
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _per_item(value, n: int, name: str, broadcast_types=(str, type(None))):
        """One value per text: a string/None is broadcast, a sequence must match."""
        if isinstance(value, broadcast_types):
            return [value] * n
        value = list(value)
        if len(value) != n:
            raise ValueError(
                f"{name} has {len(value)} entries for {n} texts; pass one per "
                f"text, or a single value for all."
            )
        return value

    def decode(self, text: str, lang: str = None, hints=None,
               aligned: bool = True, case_endings: bool = True) -> str:
        """
        Restore diacritics in one text.

        Args:
            text: Input text of any length; chunked at ~300 characters.
            lang: ISO-639-3 / ISO-639-1 code, ``"ara-nocase"`` or ``"auto"``.
                  ``None`` (default) means ``"auto"``.
            hints: Meaning hints for words whose marks depend on meaning — a
                   ``{word: meaning}`` mapping, a list of ``"word=meaning"``
                   strings, or a ready-made ``"[g: ...]"`` block. Written in
                   English; applied to every chunk of this text.
            aligned: Keep your letters, take only the model's marks (default).
            case_endings: Arabic only. ``False`` selects ``<ara-nocase>``.

        Returns:
            str: the restored text. Surrounding whitespace is trimmed, and empty
            input gives ``""``.
        """
        return self.decode_batch(
            [text], lang=lang,
            hints=None if hints is None else [hints],
            aligned=aligned, case_endings=case_endings)[0]

    def decode_batch(self, texts, lang=None, hints=None, aligned: bool = True,
                     case_endings: bool = True, batch_size: int = None) -> list:
        """
        Restore diacritics in many texts.

        Args:
            texts: Sequence of strings.
            lang: One code for all texts, or a sequence with one per text.
            hints: ``None``, or a sequence with one entry per text (each ``None``
                   or anything :func:`olaverse.nlp.diacnet_utils.build_hint`
                   accepts). Not broadcast: a list of hint strings is ambiguous
                   between "several hints" and "one per text".
            aligned: Keep your letters, take only the model's marks (default).
            case_endings: Arabic only. ``False`` selects ``<ara-nocase>``.
            batch_size: Override the instance default for this call.

        Returns:
            list[str]: one restored text per input, in order.
        """
        import torch
        import unicodedata

        from olaverse.nlp.diacnet_utils import (
            align, build_hint, chunk_text, resolve_tag)

        if isinstance(texts, str):
            raise TypeError(
                "decode_batch() takes a sequence of texts, not a single string "
                "(it would be split into characters); use decode() for one text."
            )
        texts = list(texts)
        n = len(texts)
        langs = self._per_item(lang, n, "lang")
        hint_list = [None] * n if hints is None else list(hints)
        if len(hint_list) != n:
            raise ValueError(
                f"hints has {len(hint_list)} entries for {n} texts; pass one "
                f"per text (None where there is no hint)."
            )

        results = [""] * n
        parts = {}
        items = []                       # (text index, chunk index, chunk, prompt)
        for i, text in enumerate(texts):
            text = (text or "").strip()
            if not text:
                continue
            tag = resolve_tag(langs[i], case_endings)
            hint = build_hint(hint_list[i])
            pieces = chunk_text(text, self.CHUNK_CHARS)
            parts[i] = [None] * len(pieces)
            for j, piece in enumerate(pieces):
                prompt = f"<{tag}> {hint + ' ' if hint else ''}{piece}"
                items.append((i, j, piece, prompt))

        # Longest first: batches pad to their longest member, and an
        # out-of-memory on the biggest batch shows up immediately.
        items.sort(key=lambda it: len(it[3].encode("utf-8")), reverse=True)
        step = max(1, int(batch_size or self.batch_size))
        for b0 in range(0, len(items), step):
            batch = items[b0:b0 + step]
            enc = self.tokenizer([it[3] for it in batch], return_tensors="pt",
                                 padding="longest").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=2 * enc["input_ids"].shape[1] + 16,
                    num_beams=1, do_sample=False)
            decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            for (i, j, piece, _), y in zip(batch, decoded):
                y = unicodedata.normalize("NFC", y.strip())
                parts[i][j] = align(piece, y) if aligned else y

        for i, pieces in parts.items():
            results[i] = " ".join(pieces)
        return results


_DIACTAG_DEFAULT_CKPT = "ckpt_120000.pt"
# Checkpoint filename per release. A repo not listed here (a fork) keeps the
# diactag-1.0 default, as before; pass ``ckpt=`` for anything else.
_DIACTAG_CKPTS = {
    "diactag-1.0": "ckpt_120000.pt",
    "diactag-2.0": "ckpt_final.pt",
}
# int8 first: 3x faster and 4x smaller on CPU for +0.03pp DER, and compliance
# is architectural so quantisation cannot break the strip guarantee.
_DIACTAG_ONNX_NAMES = ("diactag.int8.onnx", "diactag.onnx")

# (repo, artefact, device) -> (model, LabelSpace, temperature, backend)
_DIACTAG_CACHE = {}
_DIACTAG_LEXICON_CACHE = {}


def _diactag_default_ckpt(model_name: str) -> str:
    return _DIACTAG_CKPTS.get(model_name.rsplit("/", 1)[-1], _DIACTAG_DEFAULT_CKPT)


def _diactag_fetch(repo_id: str, filename: str, required: bool = True):
    """Resolve one artefact from a diactag repo through the olaverse cache."""
    try:
        return get_model_path(filename, repo_id=repo_id)
    except Exception as exc:
        if not required:
            return None
        raise RuntimeError(
            f"Could not fetch '{filename}' from '{repo_id}'. If the repository "
            f"is private or gated, authenticate first — either `huggingface-cli "
            f"login` or HF_TOKEN=<token with read access>. "
            f"Original error: {exc}"
        ) from exc


class OnnxTaggerSession:
    """
    Adapts an ONNX Runtime session to the ``DiacTagger`` interface, so the ONNX
    path runs through the identical decoder — same windowing, legality masking
    and invariant check — rather than a parallel implementation that could drift
    away from the PyTorch one.

    Inputs are fed and outputs read **by name**, against what the loaded graph
    declares, so one decoder serves both the current export
    (``ids/lang/lang_known/attn`` -> ``shape/tone/lid_logits``) and the earlier
    one that omitted the LID head, which is still what a pinned ``revision=``
    resolves to.

    ``lang_known`` must be a graph *input* before the LID head can be trusted.
    The language embedding is additive at every position and leaks into the
    mean-pooled state the head reads, so under "language is known" the head
    merely echoes the caller's own guess — on Polish text with ``lang=yor`` it
    answers ``yor`` at 0.9477. Both conditions are therefore required before
    ``has_lid`` is set.
    """

    def __init__(self, session, n_langs: int):
        self.sess = session
        self.n_langs = n_langs
        self.inputs = [i.name for i in session.get_inputs()]
        self.outputs = [o.name for o in session.get_outputs()]
        self.has_lid = ("lid_logits" in self.outputs
                        and "lang_known" in self.inputs)

    # DiacTagger is an nn.Module; these make the duck-type complete.
    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def __call__(self, ids, lang, lang_known=None, attn=None, need_mlm=False):
        import numpy as np
        import torch

        if attn is None:
            attn = torch.ones_like(ids, dtype=torch.bool)
        if lang_known is None:
            lang_known = torch.ones_like(lang)
        feed = {
            "ids": ids.cpu().numpy().astype(np.int64),
            "lang": lang.cpu().numpy().astype(np.int64),
            "attn": attn.cpu().numpy(),
            "lang_known": lang_known.cpu().numpy().astype(np.int64),
        }
        out = self.sess.run(
            None, {k: v for k, v in feed.items() if k in self.inputs})
        by_name = dict(zip(self.outputs, out))
        return {
            "shape": torch.from_numpy(by_name["shape_logits"]),
            "tone": torch.from_numpy(by_name["tone_logits"]),
            # The zeros are never consumed: detect_language() and a lang=None
            # restore() both raise when has_lid is False, rather than
            # arg-maxing a constant into "yor".
            "lid": (torch.from_numpy(by_name["lid_logits"]) if self.has_lid
                    else torch.zeros(ids.shape[0], self.n_langs)),
        }


class DiacTagDecoder:
    """
    diactag diacritic restoration (per-character tagger) — diactag-1.0 covers 10
    languages, diactag-2.0 adds Arabic (11).

    Which vendored code decodes a checkpoint is decided by the ``spec_version`` in
    its ``labels.json`` (1.x -> spec 1.2.0, 2.x -> spec 2.0.0), so diactag-1.0
    keeps running on the code it was built with.

    Unlike the seq2seq ``diacnet`` line, this model classifies each character
    into a diacritic transformation instead of generating output text. It has no
    mechanism for changing, inserting or deleting a base character, so::

        strip_diacritics(output) == strip_diacritics(input)

    holds by construction. The invariant is asserted on every call rather than
    assumed, and it survives int8 quantisation because it is a property of the
    architecture, not of numeric precision.

    What that buys over ``diacnet-1.1`` (or, with output alignment off, over any
    generative model): no text corruption, per-character
    calibrated confidence, built-in language detection, and 37.6M parameters
    against 580M — CPU serving is the default rather than a compromise. What it
    costs: the model cannot fix a typo, because fixing one would mean inserting
    or deleting a character.

    Args:
        model_name: Hugging Face repo id.
        ckpt: Checkpoint filename in the repo. ``None`` (default) picks the
                release's own: ``ckpt_120000.pt`` for diactag-1.0, ``ckpt_final.pt``
                for diactag-2.0. Ignored when ``onnx=True``.
        device: ``"cpu"`` (default), ``"cuda"``, ``"mps"``. Ignored when
                ``onnx=True`` — the ONNX session is CPU-only.
        min_confidence: Abstention threshold in [0, 1]. Characters the model is
                less sure about than this are left exactly as the caller typed
                them. ``0`` (default) commits to every character. Overridable
                per call.
        use_lexicon: Rerank predicted non-words against attested spellings of
                the same stripped form. Conservative — it only ever chooses
                among forms seen in the corpus. **Measure it on your data
                before enabling it:** on diacbench it cuts non-word outputs by
                27% but raises Yoruba DER by 15% (0.0836 -> 0.0961), because
                the density gating that keeps the lexicon clean shrank the
                Yoruba vocabulary to 18,436 forms, so "not in the lexicon"
                often means "rare or inflected word we didn't keep" and correct
                output gets overwritten. Off by default.
        case_endings: Arabic only (diactag-2.0). ``False`` leaves each word's last
                letter without its vowel / tanwīn / sukūn (shadda kept) — how most
                modern Arabic is vowelled. Raises on diactag-1.0, which has no
                Arabic. Overridable per call.
        onnx: Load the int8 ONNX export instead of the PyTorch checkpoint —
                3x faster and 4x smaller on CPU for +0.03pp DER, and the strip
                guarantee survives quantisation because it is architectural.
                Language auto-detection works here too, matching the PyTorch
                head; against an export predating the LID head, ``lang``
                becomes required rather than silently guessed. Requires
                ``onnxruntime``.
    """

    #: Language codes accepted by :meth:`decode`, ISO-639-3 (ISO-639-1 also
    #: works). This is the diactag-1.0 set; an instance narrows or widens it to
    #: what its own label space covers (diactag-2.0 adds ``ara``).
    LANGUAGES = ("yor", "ibo", "hau", "vie", "pol", "tur", "por", "spa",
                 "fra", "ita")

    # Set on construction. Class-level so a bare ``__new__`` instance (as the
    # language-validation tests build) still resolves against diactag-1.0.
    _backend = None

    def __init__(self, model_name: str = "olaverse/diactag-1.0",
                 ckpt: str = None, device: str = "cpu",
                 min_confidence: float = 0.0, use_lexicon: bool = False,
                 onnx: bool = False, case_endings: bool = True):
        self.model_name = model_name
        self.device = "cpu" if onnx else device
        self.min_confidence = min_confidence
        self.onnx = onnx
        self.case_endings = case_endings
        if ckpt is None:
            ckpt = _diactag_default_ckpt(model_name)

        key = (model_name, "onnx" if onnx else ckpt, self.device)
        if key not in _DIACTAG_CACHE:
            _DIACTAG_CACHE[key] = self._load(model_name, ckpt, self.device, onnx)
        (self._model, self._labels, self._temperature,
         self._backend) = _DIACTAG_CACHE[key]
        self.LANGUAGES = self._backend.languages
        if not case_endings:
            self._require_case_endings()

        # Read the capability off the graph rather than assuming it from
        # `onnx`. The current export carries the LID head, but exports before
        # it returned SHAPE and TONE only, and auto-detection against one of
        # those silently answers "yor" for every input.
        self.supports_language_detection = (
            self._model.has_lid if onnx else True)

        self._lexicon = (self._load_lexicon(model_name, self._backend)
                         if use_lexicon else None)
        # One runtime per abstention threshold. The expensive parts (weights,
        # label space, lexicon) are shared; a runtime is just a config plus the
        # legality mask, so a caller can move along the coverage curve without
        # reloading anything.
        self._runtimes = {}

    # -- loading ----------------------------------------------------------
    @staticmethod
    def _load(model_name, ckpt, device, onnx):
        import json as _json

        from olaverse.nlp._diactag_backend import backend_for_labels

        # The label space is versioned: the spec in labels.json decides which
        # vendored code decodes this checkpoint (1.x -> _diactag, 2.x ->
        # _diactag2). LabelSpace.load then requires an exact spec match.
        labels_path = _diactag_fetch(model_name, "labels.json")
        backend = backend_for_labels(labels_path)
        labels = backend.module("labels").LabelSpace.load(labels_path)

        temperature = 1.0
        calibration = _diactag_fetch(model_name, "calibration.json", required=False)
        if calibration:
            try:
                with open(calibration, encoding="utf-8") as f:
                    temperature = _json.load(f).get("shared", 1.0)
            except Exception:
                pass

        if onnx:
            model = DiacTagDecoder._load_onnx(model_name, labels)
        else:
            DiacTagger = backend.module("model").DiacTagger
            model, _ = DiacTagger.load(_diactag_fetch(model_name, ckpt),
                                       map_location=device)
        return model, labels, temperature, backend

    @staticmethod
    def _load_onnx(model_name, labels):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnx=True requires onnxruntime. Install it with "
                "`pip install olaverse[onnx]`."
            ) from exc

        path = None
        for name in _DIACTAG_ONNX_NAMES:
            path = _diactag_fetch(model_name, name, required=False)
            if path:
                break
        if path is None:
            raise FileNotFoundError(f"No ONNX export found in '{model_name}'.")

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(path, so,
                                       providers=["CPUExecutionProvider"])
        return OnnxTaggerSession(session, labels.n_langs)

    @staticmethod
    def _load_lexicon(model_name, backend):
        # The lexicon is keyed through the backend's unicode_ops, so it is
        # loaded by the same generation of code as the checkpoint.
        Lexicon = backend.module("lexicon").Lexicon
        key = (model_name, backend.package)
        if key not in _DIACTAG_LEXICON_CACHE:
            # Raises if the repo has no lexicon. Silently returning the plain
            # model would leave use_lexicon=True doing nothing at all.
            _DIACTAG_LEXICON_CACHE[key] = Lexicon.load(
                _diactag_fetch(model_name, "lexicon.json"))
        return _DIACTAG_LEXICON_CACHE[key]

    def _active_backend(self):
        if self._backend is None:
            from olaverse.nlp._diactag_backend import backend_for_spec
            return backend_for_spec("1")        # diactag-1.0, the default model
        return self._backend

    def _model_label(self):
        return getattr(self, "model_name", "olaverse/diactag-1.0").rsplit("/", 1)[-1]

    def _require_case_endings(self):
        if not self._active_backend().supports_case_endings:
            raise ValueError(
                f"case_endings only applies to Arabic, which '{self._model_label()}' "
                f"does not cover. Use diactag-2.0, or leave case_endings at its "
                f"default."
            )

    def _runtime(self, min_confidence, case_endings=None):
        threshold = (self.min_confidence if min_confidence is None
                     else float(min_confidence))
        endings = self.case_endings if case_endings is None else bool(case_endings)
        if not endings:
            self._require_case_endings()
        key = (threshold, endings)
        if key not in self._runtimes:
            infer = self._backend.module("infer")
            kwargs = {}
            if self._backend.supports_case_endings:
                kwargs["case_endings"] = endings
            cfg = infer.InferConfig(
                device=self.device,
                temperature=self._temperature,
                min_confidence=threshold,
                use_legality=True,
                lexicon_mode="rerank" if self._lexicon else "off",
                **kwargs,
            )
            self._runtimes[key] = infer.Diacritizer(
                self._model, self._labels, cfg, self._lexicon)
        return self._runtimes[key]

    # -- inference --------------------------------------------------------
    def normalize_language(self, lang):
        """
        Map a language code to the ISO-639-3 form the model uses, raising on
        anything it does not support. ``None`` passes through and means
        "detect it".
        """
        if lang is None:
            return None
        backend = self._active_backend()
        resolved = backend.module("unicode_ops").normalize_lang(lang)
        if resolved is None:
            raise ValueError(
                f"Unsupported language '{lang}' for {self._model_label()}. "
                f"Supported: {list(backend.languages)} (ISO-639-1 codes such as "
                f"'yo' are also accepted). Pass lang=None to auto-detect."
            )
        return resolved

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Identify the language of ``text`` with the model's own LID head.

        Returns:
            tuple: ``(code, probability)``, where ``code`` is ISO-639-3.
        """
        self._require_lid()
        return self._runtime(None).detect_language(text)

    def _require_lid(self):
        if not self.supports_language_detection:
            raise ValueError(
                "This ONNX export does not include the language-detection "
                "head, so language cannot be auto-detected from it. Pass an "
                "explicit lang=, or construct with onnx=False to use the "
                "PyTorch checkpoint. Exports published from 2026-08-04 carry "
                "the head; a pinned older revision will not."
            )

    def decode(self, text: str, lang: str = None, min_confidence: float = None,
               return_details: bool = False,
               case_endings: bool = None) -> Union[str, Tuple[str, List]]:
        """
        Restore diacritics.

        Args:
            text: Input text. Documents are handled directly — the model slides
                  an overlapping window and keeps only the centre of each, so
                  every character is predicted with context on both sides.
            lang: ISO-639-3 or ISO-639-1 code. ``None`` (default) runs the LID
                  head and uses what it detects, which costs ~0.0001 DER.
            min_confidence: Per-call override of the abstention threshold.
            return_details: Also return a list of per-character results
                  (``char``, ``confidence``, ``abstained``, ``protected``), one
                  per grapheme, for routing low-confidence spans to review.
            case_endings: Arabic only (diactag-2.0). ``False`` drops the vowel,
                  tanwīn or sukūn on each word's last letter (shadda is kept).
                  ``None`` uses the value given at construction. Has no effect
                  on other languages.

        Returns:
            Union[str, Tuple[str, List]]: the restored text, or ``(text, details)`` when ``return_details=True``.
        """
        text = text or ""
        if not text.strip():
            return ("", []) if return_details else ""
        resolved = self.normalize_language(lang)
        if resolved is None:
            self._require_lid()
        return self._runtime(min_confidence, case_endings).restore(
            text, resolved, return_details=return_details)


class TransformerDecoder:
    def __init__(self, pt_path, vocab_path):
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            v_data = json.load(f)
        self.word_candidates = v_data["word_candidates"]
        base_model = v_data["base_model"]
        
        class DiacNetYorXModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_pretrained(base_model)
                self.classifier = nn.Linear(self.encoder.config.hidden_size, 8)
            def forward(self, input_ids, attention_mask):
                return self.classifier(self.encoder(input_ids, attention_mask).last_hidden_state)
                
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = DiacNetYorXModel().to(self.device)
        
        state = torch.load(pt_path, map_location=self.device)
        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

    def decode(self, text):
        import torch
        plain_words = re.findall(r'\S+', strip_all_diacritics(text))
        if not plain_words: return text
        
        encoding = self.tokenizer(plain_words, is_split_into_words=True, return_tensors='pt', truncation=True, max_length=128)
        input_ids = encoding['input_ids'].to(self.device)
        attn_mask = encoding['attention_mask'].to(self.device)
        word_ids = encoding.word_ids()
        
        with torch.no_grad():
            logits = self.model(input_ids, attn_mask)
            preds = logits.argmax(dim=-1).squeeze(0)
            
        corrected_words = []
        prev_wid = None
        for j, wid in enumerate(word_ids):
            if wid is None or wid == prev_wid: continue
            prev_wid = wid
            pw = plain_words[wid].lower()
            cands = self.word_candidates.get(pw, [pw])
            pred_idx = preds[j].item()
            if pred_idx < len(cands):
                pred_label = cands[pred_idx]
            else:
                pred_label = cands[0]
                
            if plain_words[wid].isupper():
                pred_label = pred_label.upper()
            elif plain_words[wid][0].isupper():
                pred_label = pred_label.capitalize()
            corrected_words.append(pred_label)
            
        tokens = re.findall(r'[\w\u0300-\u036f]+|[^\w\s]|\s+', text)
        word_indices = [i for i, t in enumerate(tokens) if t.strip() and re.match(r'^[\w\u0300-\u036f]+$', t)]
        for w_idx, corrected in zip(word_indices, corrected_words):
            tokens[w_idx] = corrected
            
        return "".join(tokens)

# ─── Legacy Functions ───
def diacritize_yoruba(text, model_path=None):
    global _YORUBA_MODEL_CACHE
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("yoruba_diacritizer.json", repo_id="olaverse/diacnet-yor-viterbi")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yoruba_diacritizer.json")
    if resolved_path in _YORUBA_MODEL_CACHE:
        model = _YORUBA_MODEL_CACHE[resolved_path]
    else:
        model = _load_diacritizer_model(resolved_path, is_custom=(model_path is not None))
        _YORUBA_MODEL_CACHE[resolved_path] = model
    return viterbi_decode(text, model)

def diacritize_yoruba_dot_below(text, model_path=None):
    global _YORUBA_DB_MODEL_CACHE
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("yoruba_diacritizer_dot_below.json", repo_id="olaverse/diacnet-yor-db")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yoruba_diacritizer_dot_below.json")
    if resolved_path in _YORUBA_DB_MODEL_CACHE:
        model = _YORUBA_DB_MODEL_CACHE[resolved_path]
    else:
        model = _load_diacritizer_model(resolved_path, is_custom=(model_path is not None))
        _YORUBA_DB_MODEL_CACHE[resolved_path] = model
    return knn_decode(text, model, {'o', 'e', 's'})

def diacritize_igbo(text, model_path=None):
    global _IGBO_MODEL_CACHE
    resolved_path = model_path
    if resolved_path is None:
        try:
            resolved_path = get_model_path("igbo_diacritizer.json", repo_id="olaverse/diacnet-ig")
        except Exception:
            resolved_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "igbo_diacritizer.json")
    if resolved_path in _IGBO_MODEL_CACHE:
        model = _IGBO_MODEL_CACHE[resolved_path]
    else:
        model = _load_diacritizer_model(resolved_path, is_custom=(model_path is not None))
        _IGBO_MODEL_CACHE[resolved_path] = model
    return knn_decode(text, model, {'i', 'u', 'o', 'e'})

# ─── Unified Wrapper Class ───
MODEL_REGISTRY = {
    "diacnet-yor-viterbi": {"lang": "yo", "method": "viterbi"},
    "diacnet-yor-db":      {"lang": "yo", "method": "knn"},
    "diacnet-ig":          {"lang": "ig", "method": "knn"},
    "diacnet-yor":         {"lang": "yo", "method": "bilstm"},
    "diacnet-yor-x":       {"lang": "yo", "method": "transformer"},
    "diacnet-1.0":         {"lang": "multi", "method": "diacnet"},
    "diacnet-1.1":         {"lang": "multi", "method": "diacnet"},
    "diactag-1.0":         {"lang": "multi", "method": "diactag"},
    "diactag-2.0":         {"lang": "multi", "method": "diactag"},
    "diacnet-2.0":         {"lang": "multi", "method": "diacnet2"},
    "diacnet-mini-2.0":    {"lang": "multi", "method": "diacnet2"},
    "auto":                {"lang": "auto", "method": "auto"},
}

class Diacritizer:
    """
    Unified interface for restoring diacritics in African languages.

    Pass a model ID to use a specific backend, or ``model="auto"`` to detect
    the language automatically and route to the appropriate diacritizer.

    Args:
        model: One of:

            * ``"diacnet-yor-viterbi"`` — Yoruba, fast Viterbi n-gram (default)
            * ``"diacnet-yor-db"``      — Yoruba dot-below only, KNN
            * ``"diacnet-ig"``          — Igbo, KNN
            * ``"diacnet-yor"``         — Yoruba BiLSTM (requires ``olaverse[deeplearning]``)
            * ``"diacnet-yor-x"``       — Yoruba XLM-RoBERTa (requires ``olaverse[deeplearning]``)
            * ``"diacnet-1.0"``         — Multilingual DiacNet, 10 languages, see ``lang=``
                                          (requires ``olaverse[deeplearning]``)
            * ``"diacnet-1.1"``         — Same architecture, larger corpus. Better on
                                          vie/tur/pol/ita/por, worse on yor/ibo/hau
                                          (requires ``olaverse[deeplearning]``)
            * ``"diactag-1.0"``         — Per-character tagger, 10 languages. Cannot
                                          corrupt the text, 38MB on CPU, best DER on
                                          7 of 10 languages (requires ``olaverse[deeplearning]``)
            * ``"diactag-2.0"``         — Same tagger architecture, 11 languages: adds
                                          Arabic (``ara``/``ar``) with a no-case-endings
                                          mode. Matches 1.0 on the other ten
                                          (requires ``olaverse[deeplearning]``)
            * ``"diacnet-2.0"``         — ByT5 text-to-text, 11 languages incl. Arabic,
                                          582M. Meaning hints, ``<auto>`` language, output
                                          alignment on by default
                                          (requires ``olaverse[deeplearning]``)
            * ``"diacnet-mini-2.0"``    — Same interface, 300M, close to diacnet-2.0
                                          (requires ``olaverse[deeplearning]``)
            * ``"auto"``                — detect language via LIDLite5, then route automatically

        lang: Target language for the multilingual models. One of
              ``"yo", "vi", "ig", "ha", "pl", "tr", "pt", "es", "fr", "it"``, or the
              ISO-639-3 equivalent for the diactag and diacnet-2.x models; those
              also take ``"ar"``/``"ara"``, and diacnet-2.x additionally
              ``"ara-nocase"`` and ``"auto"``. Ignored by the single-language
              models. For the diactag models leaving it ``None`` auto-detects
              with the model's own LID head, and diacnet-2.x sends ``<auto>``;
              ``"diacnet-1.0"``/``"1.1"`` fall back to Yoruba.

        split_sentences: ``diacnet-1.0``/``1.1`` only. They were trained on
              sentence-length input, so multi-sentence text is segmented and
              restored a sentence at a time by default. Set ``False`` to send the
              whole string through in one pass. The diactag models handle documents
              natively with sliding windows, and diacnet-2.x chunks at ~300
              characters instead; both ignore this.

        splitter: ``diacnet-1.0``/``1.1`` only. Your own callable taking a string and
              returning a list of segments, replacing the default sentence
              splitter.

        min_confidence: diactag models only. Abstention threshold in [0, 1].
              Characters the model is less sure about are left exactly as the
              caller typed them. At 0.9, ~97% of characters are restored at
              99.6% accuracy and the rest are flagged. Default 0 commits to
              everything. Overridable per :meth:`restore` call.

        use_lexicon: diactag models only. Rerank predicted non-words against
              attested spellings of the same stripped form. Off by default, and
              worth measuring before you turn it on — on diacbench it cuts
              non-word outputs by 27% but raises Yoruba DER by 15%
              (0.0836 -> 0.0961). See :class:`DiacTagDecoder`.

        onnx: diactag models only. Load the int8 ONNX export — 3x faster and
              4x smaller on CPU for +0.03pp DER, with language auto-detection
              intact. Requires ``olaverse[onnx]``.

        device: ``"diactag-*"`` and ``"diacnet-2.0"``/``"diacnet-mini-2.0"``.
              ``"cpu"`` (default), ``"cuda"`` or ``"mps"``; ``"auto"`` picks CUDA
              when available. Ignored by diactag when ``onnx=True``. The diacnet-2.x
              models load in bfloat16 on CUDA and float32 elsewhere.

        case_endings: Arabic only (``"diactag-2.0"``, ``"diacnet-2.0"``,
              ``"diacnet-mini-2.0"``). ``False`` leaves each word's last letter
              without its vowel / tanwīn / sukūn (shadda kept), which is how most
              modern Arabic is vowelled. Has no effect on other languages.
              Overridable per :meth:`restore` call.

        aligned: ``diacnet`` models only. Keep your own letters and take only the
              model's marks, so the output always strips back to the input.
              Defaults to ``True`` for ``"diacnet-2.0"``/``"diacnet-mini-2.0"`` and
              ``False`` for ``"diacnet-1.0"``/``"1.1"``, whose output is therefore
              unchanged unless you ask. Overridable per :meth:`restore` call.

        batch_size: ``"diacnet-2.0"``/``"diacnet-mini-2.0"`` only. Chunks decoded
              together by :meth:`restore_batch`.
    """

    def __init__(self, model: str = "diacnet-yor-viterbi", lang: str = None,
                 split_sentences: bool = True, splitter: "callable" = None,
                 min_confidence: float = 0.0, use_lexicon: bool = False,
                 onnx: bool = False, device: str = "cpu",
                 case_endings: bool = True, aligned: bool = None,
                 batch_size: int = 16):
        if model not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model}' is not recognised. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        config = MODEL_REGISTRY[model]
        self.lang = config["lang"]
        self.method = config["method"]
        self.neural_decoder = None
        self.diacnet_lang = lang or "yo"
        # diactag treats "no language given" as "detect it", so the raw value is
        # kept rather than defaulted to Yoruba.
        self.diactag_lang = lang
        self.split_sentences = split_sentences
        self.splitter = splitter
        self.case_endings = case_endings
        # diacnet-2.x aligns by default; 1.x keeps its historical raw output.
        self.aligned = (self.method == "diacnet2") if aligned is None else aligned
        if not case_endings and self.method not in ("diactag", "diacnet2"):
            raise ValueError(
                f"case_endings is only supported by the Arabic-capable models "
                f"(diactag-2.0, diacnet-2.0, diacnet-mini-2.0); this Diacritizer "
                f"is using '{model}'."
            )
        if aligned is not None and self.method not in ("diacnet", "diacnet2"):
            raise ValueError(
                f"aligned is only supported by the diacnet models; this "
                f"Diacritizer is using '{model}'."
            )

        # Auto-routing: lazy-load LIDLite5 + sub-diacritizers at restore() time
        if self.method == "auto":
            self._lid = None
            self._sub: dict = {}
            return

        if self.method == "bilstm":
            if "bilstm" not in _NEURAL_CACHE:
                pt    = get_model_path("diacnet_yor.pt",       repo_id="olaverse/diacnet-yor")
                vocab = get_model_path("diacnet_yor_vocab.json", repo_id="olaverse/diacnet-yor")
                _NEURAL_CACHE["bilstm"] = BiLSTMDecoder(pt, vocab)
            self.neural_decoder = _NEURAL_CACHE["bilstm"]

        elif self.method == "transformer":
            if "transformer" not in _NEURAL_CACHE:
                pt    = get_model_path("diacnet_yor_x.pt",       repo_id="olaverse/diacnet-yor-x")
                vocab = get_model_path("diacnet_yor_x_vocab.json", repo_id="olaverse/diacnet-yor-x")
                _NEURAL_CACHE["transformer"] = TransformerDecoder(pt, vocab)
            self.neural_decoder = _NEURAL_CACHE["transformer"]

        elif self.method == "diacnet":
            # Cache per version key ("diacnet-1.0", "diacnet-1.1", ...) so
            # future DiacNet releases are one registry line, same decoder class.
            if model not in _NEURAL_CACHE:
                _NEURAL_CACHE[model] = DiacNetDecoder(f"olaverse/{model}")
            self.neural_decoder = _NEURAL_CACHE[model]

        elif self.method == "diactag":
            # DiacTagDecoder does its own caching of the weights and label
            # space, so each instance is cheap even though the wrapper is not
            # shared: min_confidence, lexicon and device are per-instance.
            self.neural_decoder = DiacTagDecoder(
                model_name=f"olaverse/{model}",
                device=device,
                min_confidence=min_confidence,
                use_lexicon=use_lexicon,
                onnx=onnx,
                case_endings=case_endings,
            )
            # Fail on an unsupported code now rather than silently
            # auto-detecting on the first restore() call.
            self.diactag_lang = self.neural_decoder.normalize_language(lang)

        elif self.method == "diacnet2":
            from olaverse.nlp.diacnet_utils import resolve_tag
            # Fail on an unsupported code now, before a 2GB download.
            resolve_tag(lang, case_endings)
            self.batch_size = int(batch_size)
            # Weights are shared per (model, device); batch_size is passed per
            # call so changing it never reloads or mutates the shared decoder.
            key = (model, device)
            if key not in _NEURAL_CACHE:
                _NEURAL_CACHE[key] = DiacNet2Decoder(
                    f"olaverse/{model}", device=device)
            self.neural_decoder = _NEURAL_CACHE[key]

    def _auto_restore(self, text: str) -> str:
        """Detect language then delegate to the correct diacritizer."""
        if self._lid is None:
            from olaverse.nlp.language_detection import LIDLite5
            self._lid = LIDLite5()

        lang = self._lid.predict(text)

        if lang == "ibo":
            if "ig" not in self._sub:
                self._sub["ig"] = Diacritizer(model="diacnet-ig")
            return self._sub["ig"].restore(text)

        # Default to Yoruba for 'yor' and any other detected language
        if "yo" not in self._sub:
            self._sub["yo"] = Diacritizer(model="diacnet-yor-viterbi")
        return self._sub["yo"].restore(text)

    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Identify the language of ``text``. The diactag models only — they are the
        only ones with a language-identification head of their own.

        Returns:
            tuple: ``(iso_639_3_code, probability)``.
        """
        if self.method != "diactag":
            raise ValueError(
                f"detect_language() is only available on the diactag models; this "
                f"Diacritizer is using '{self.method}'. For standalone language "
                f"identification use olaverse.nlp.LIDLite5 / LIDNeural25."
            )
        return self.neural_decoder.detect_language(text)

    def restore(self, text: str, lang: str = None, min_confidence: float = None,
                return_details: bool = False, case_endings: bool = None,
                aligned: bool = None, hints=None) -> Union[str, Tuple[str, List]]:
        """
        Restore diacritics in the given text.

        Args:
            text: Plain text (tones/diacritics stripped or missing).
            lang: Per-call language override for the multilingual models,
                  replacing the one given at construction.
            min_confidence: ``"diactag-*"`` only. Per-call abstention
                  threshold, so one loaded model can serve a CMS pre-fill and a
                  legal pipeline at different points on the coverage curve.
            return_details: ``"diactag-*"`` only. Also return per-character
                  results (``char``, ``confidence``, ``abstained``,
                  ``protected``) for routing low-confidence spans to review.
            case_endings: Arabic only (``"diactag-2.0"``, ``"diacnet-2.0"``,
                  ``"diacnet-mini-2.0"``). Per-call override of the constructor
                  setting; ``False`` drops the vowel on each word's last letter.
            aligned: ``diacnet`` models only. Per-call override of the
                  constructor setting: keep your letters, take only the marks.
            hints: ``"diacnet-2.0"``/``"diacnet-mini-2.0"`` only. Meaning hints
                  for words whose marks depend on meaning: a ``{word: meaning}``
                  mapping, a list of ``"word=meaning"`` strings, or a ready-made
                  ``"[g: ...]"`` block. Written in English.

        Returns:
            Union[str, Tuple[str, List]]: the restored text, or ``(text, details)`` when ``return_details=True``.
        """
        if return_details and self.method != "diactag":
            raise ValueError(
                f"return_details=True is only supported by the diactag models, "
                f"which score each character independently; this Diacritizer is "
                f"using '{self.method}'."
            )
        if min_confidence is not None and self.method != "diactag":
            raise ValueError(
                f"min_confidence is only supported by the diactag models; this "
                f"Diacritizer is using '{self.method}'."
            )
        if case_endings is not None and self.method not in ("diactag", "diacnet2"):
            raise ValueError(
                f"case_endings is only supported by the Arabic-capable models "
                f"(diactag-2.0, diacnet-2.0, diacnet-mini-2.0); this Diacritizer "
                f"is using '{self.method}'."
            )
        if aligned is not None and self.method not in ("diacnet", "diacnet2"):
            raise ValueError(
                f"aligned is only supported by the diacnet models; this "
                f"Diacritizer is using '{self.method}'."
            )
        if hints is not None and self.method != "diacnet2":
            raise ValueError(
                f"hints is only supported by diacnet-2.0 and diacnet-mini-2.0; "
                f"this Diacritizer is using '{self.method}'."
            )

        if self.method == "diactag":
            return self.neural_decoder.decode(
                text,
                lang=lang if lang is not None else self.diactag_lang,
                min_confidence=min_confidence,
                return_details=return_details,
                case_endings=case_endings,
            )

        if self.method == "auto":
            return self._auto_restore(text)

        if self.method == "diacnet":
            return self.neural_decoder.decode(
                text,
                lang=lang or self.diacnet_lang,
                split_sentences=self.split_sentences,
                splitter=self.splitter,
                aligned=self.aligned if aligned is None else aligned,
            )

        if self.method == "diacnet2":
            return self.restore_batch(
                [text], lang=lang, case_endings=case_endings, aligned=aligned,
                hints=None if hints is None else [hints])[0]

        if lang is not None:
            raise ValueError(
                f"lang= is only meaningful for the multilingual models "
                f"(diacnet-1.0, diacnet-1.1, diactag-1.0); this Diacritizer is "
                f"using '{self.method}', which is single-language."
            )

        if self.neural_decoder:
            return self.neural_decoder.decode(text)

        if self.lang == "yo":
            if self.method == "viterbi":
                return diacritize_yoruba(text)
            elif self.method == "knn":
                return diacritize_yoruba_dot_below(text)
            else:
                raise ValueError(f"Unsupported method '{self.method}' for Yoruba.")

        if self.lang == "ig":
            return diacritize_igbo(text)

        raise ValueError(f"Unsupported language '{self.lang}'.")

    def restore_batch(self, texts, lang=None, case_endings: bool = None,
                      aligned: bool = None, hints=None) -> list:
        """
        Restore diacritics in many texts.

        ``"diacnet-2.0"``/``"diacnet-mini-2.0"`` pool the ~300-character chunks of
        every text, sort them by length and decode them in padded batches, which
        is far faster on a GPU than one call per text. Every other model simply
        restores the texts one at a time.

        Args:
            texts: Sequence of strings.
            lang: One language for all texts, or a sequence with one per text.
                  Defaults to the one given at construction.
            case_endings: Arabic only; per-call override.
            aligned: ``diacnet`` models only; per-call override.
            hints: ``"diacnet-2.0"``/``"diacnet-mini-2.0"`` only. A sequence with
                  one entry per text (``None`` where there is no hint).

        Returns:
            list[str]: one restored text per input, in order.
        """
        if isinstance(texts, str):
            raise TypeError(
                "restore_batch() takes a sequence of texts, not a single string; "
                "use restore() for one text."
            )
        texts = list(texts)
        if self.method != "diacnet2":
            if hints is not None:
                raise ValueError(
                    f"hints is only supported by diacnet-2.0 and diacnet-mini-2.0; "
                    f"this Diacritizer is using '{self.method}'."
                )
            langs = lang if isinstance(lang, (list, tuple)) else [lang] * len(texts)
            if len(langs) != len(texts):
                raise ValueError(
                    f"lang has {len(langs)} entries for {len(texts)} texts.")
            return [self.restore(t, lang=l, case_endings=case_endings, aligned=aligned)
                    for t, l in zip(texts, langs)]

        if case_endings is None:
            case_endings = self.case_endings
        return self.neural_decoder.decode_batch(
            texts,
            lang=lang if lang is not None else self.diactag_lang,
            hints=hints,
            aligned=self.aligned if aligned is None else aligned,
            case_endings=case_endings,
            batch_size=self.batch_size,
        )
