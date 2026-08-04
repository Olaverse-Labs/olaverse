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

The two multilingual families differ in kind, not degree. A seq2seq model
generates the output text, so it *can* drop a word or rewrite a clause; on
Hausa only 94.7% of diacnet-1.1's outputs still stripped back to their input.
The tagger classifies each character into a diacritic transformation and copies
the base character, so ``strip(output) == strip(input)`` holds by construction
and is asserted on every call. The price is that a tagger cannot insert or
delete characters, so it cannot fix a typo. Prefer diactag-1.0 unless you need
that, or unless you are on Vietnamese or Portuguese, where diacnet still wins.
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
               split_sentences: bool = True, splitter=None) -> str:
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
            return self._decode_one(tag, text, max_new_tokens)

        segments = (splitter or self.splitter)(text)
        if len(segments) <= 1:
            return self._decode_one(tag, text, max_new_tokens)
        return " ".join(self._decode_one(tag, s, max_new_tokens) for s in segments)

    def _decode_one(self, tag: str, text: str, max_new_tokens: int) -> str:
        import torch

        inputs = self.tokenizer(
            f"{tag} {text.strip()}",
            return_tensors="pt",
            truncation=True,
            max_length=self.MAX_INPUT_TOKENS,
        )
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

_DIACTAG_DEFAULT_CKPT = "ckpt_120000.pt"
# int8 first: 3x faster and 4x smaller on CPU for +0.03pp DER, and compliance
# is architectural so quantisation cannot break the strip guarantee.
_DIACTAG_ONNX_NAMES = ("diactag.int8.onnx", "diactag.onnx")

# (repo, artefact, device) -> (model, LabelSpace, temperature)
_DIACTAG_CACHE = {}
_DIACTAG_LEXICON_CACHE = {}


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
    diactag-1.0 diacritic restoration (per-character tagger) — 10 languages.

    Unlike the seq2seq ``diacnet`` line, this model classifies each character
    into a diacritic transformation instead of generating output text. It has no
    mechanism for changing, inserting or deleting a base character, so::

        strip_diacritics(output) == strip_diacritics(input)

    holds by construction. The invariant is asserted on every call rather than
    assumed, and it survives int8 quantisation because it is a property of the
    architecture, not of numeric precision.

    What that buys over ``diacnet-1.1``: no text corruption, per-character
    calibrated confidence, built-in language detection, and 37.6M parameters
    against 580M — CPU serving is the default rather than a compromise. What it
    costs: the model cannot fix a typo, because fixing one would mean inserting
    or deleting a character.

    Args:
        model_name: Hugging Face repo id.
        ckpt: Checkpoint filename in the repo. Ignored when ``onnx=True``.
        device: ``"cpu"`` (default), ``"cuda"``, ``"mps"``. Ignored when
                ``onnx=True`` — the ONNX session is CPU-only.
        min_confidence: Abstention threshold in [0, 1]. Characters the model is
                less sure about than this are left exactly as the caller typed
                them. ``0`` (default) commits to every character. Overridable
                per call.
        use_lexicon: Rerank predicted non-words against attested spellings of
                the same stripped form. Conservative — it only ever chooses
                among forms seen in the corpus.
        onnx: Load the int8 ONNX export instead of the PyTorch checkpoint —
                3x faster and 4x smaller on CPU for +0.03pp DER, and the strip
                guarantee survives quantisation because it is architectural.
                Language auto-detection works here too, matching the PyTorch
                head; against an export predating the LID head, ``lang``
                becomes required rather than silently guessed. Requires
                ``onnxruntime``.
    """

    #: Language codes accepted by :meth:`decode`, ISO-639-3 and ISO-639-1.
    LANGUAGES = ("yor", "ibo", "hau", "vie", "pol", "tur", "por", "spa",
                 "fra", "ita")

    def __init__(self, model_name: str = "olaverse/diactag-1.0",
                 ckpt: str = _DIACTAG_DEFAULT_CKPT, device: str = "cpu",
                 min_confidence: float = 0.0, use_lexicon: bool = False,
                 onnx: bool = False):
        self.model_name = model_name
        self.device = "cpu" if onnx else device
        self.min_confidence = min_confidence
        self.onnx = onnx

        key = (model_name, "onnx" if onnx else ckpt, self.device)
        if key not in _DIACTAG_CACHE:
            _DIACTAG_CACHE[key] = self._load(model_name, ckpt, self.device, onnx)
        self._model, self._labels, self._temperature = _DIACTAG_CACHE[key]

        # Read the capability off the graph rather than assuming it from
        # `onnx`. The current export carries the LID head, but exports before
        # it returned SHAPE and TONE only, and auto-detection against one of
        # those silently answers "yor" for every input.
        self.supports_language_detection = (
            self._model.has_lid if onnx else True)

        self._lexicon = self._load_lexicon(model_name) if use_lexicon else None
        # One runtime per abstention threshold. The expensive parts (weights,
        # label space, lexicon) are shared; a runtime is just a config plus the
        # legality mask, so a caller can move along the coverage curve without
        # reloading anything.
        self._runtimes = {}

    # -- loading ----------------------------------------------------------
    @staticmethod
    def _load(model_name, ckpt, device, onnx):
        import json as _json

        from olaverse.nlp._diactag.labels import LabelSpace

        labels = LabelSpace.load(_diactag_fetch(model_name, "labels.json"))

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
            from olaverse.nlp._diactag.model import DiacTagger
            model, _ = DiacTagger.load(_diactag_fetch(model_name, ckpt),
                                       map_location=device)
        return model, labels, temperature

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
    def _load_lexicon(model_name):
        from olaverse.nlp._diactag.lexicon import Lexicon
        if model_name not in _DIACTAG_LEXICON_CACHE:
            # Raises if the repo has no lexicon. Silently returning the plain
            # model would leave use_lexicon=True doing nothing at all.
            _DIACTAG_LEXICON_CACHE[model_name] = Lexicon.load(
                _diactag_fetch(model_name, "lexicon.json"))
        return _DIACTAG_LEXICON_CACHE[model_name]

    def _runtime(self, min_confidence):
        threshold = (self.min_confidence if min_confidence is None
                     else float(min_confidence))
        if threshold not in self._runtimes:
            from olaverse.nlp._diactag.infer import (
                Diacritizer as _Runtime, InferConfig)
            cfg = InferConfig(
                device=self.device,
                temperature=self._temperature,
                min_confidence=threshold,
                use_legality=True,
                lexicon_mode="rerank" if self._lexicon else "off",
            )
            self._runtimes[threshold] = _Runtime(
                self._model, self._labels, cfg, self._lexicon)
        return self._runtimes[threshold]

    # -- inference --------------------------------------------------------
    def normalize_language(self, lang):
        """
        Map a language code to the ISO-639-3 form the model uses, raising on
        anything it does not support. ``None`` passes through and means
        "detect it".
        """
        if lang is None:
            return None
        from olaverse.nlp._diactag.unicode_ops import normalize_lang
        resolved = normalize_lang(lang)
        if resolved is None:
            raise ValueError(
                f"Unsupported language '{lang}' for diactag-1.0. Supported: "
                f"{list(self.LANGUAGES)} (ISO-639-1 codes such as 'yo' are also "
                f"accepted). Pass lang=None to auto-detect."
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
               return_details: bool = False) -> Union[str, Tuple[str, List]]:
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

        Returns:
            Union[str, Tuple[str, List]]: the restored text, or ``(text, details)`` when ``return_details=True``.
        """
        text = text or ""
        if not text.strip():
            return ("", []) if return_details else ""
        resolved = self.normalize_language(lang)
        if resolved is None:
            self._require_lid()
        return self._runtime(min_confidence).restore(
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
            * ``"auto"``                — detect language via LIDLite5, then route automatically

        lang: Target language for the multilingual models. One of
              ``"yo", "vi", "ig", "ha", "pl", "tr", "pt", "es", "fr", "it"``, or the
              ISO-639-3 equivalent for ``"diactag-1.0"``. Ignored by the
              single-language models. For ``"diactag-1.0"`` leaving it ``None``
              auto-detects with the model's own LID head; ``"diacnet-1.0"``/``"1.1"``
              fall back to Yoruba.

        split_sentences: ``diacnet`` models only. They were trained on
              sentence-length input, so multi-sentence text is segmented and
              restored a sentence at a time by default. Set ``False`` to send the
              whole string through in one pass. ``diactag-1.0`` handles documents
              natively with sliding windows and ignores this.

        splitter: ``diacnet`` models only. Your own callable taking a string and
              returning a list of segments, replacing the default sentence
              splitter.

        min_confidence: ``"diactag-1.0"`` only. Abstention threshold in [0, 1].
              Characters the model is less sure about are left exactly as the
              caller typed them. At 0.9, ~97% of characters are restored at
              99.6% accuracy and the rest are flagged. Default 0 commits to
              everything. Overridable per :meth:`restore` call.

        use_lexicon: ``"diactag-1.0"`` only. Rerank predicted non-words against
              attested spellings of the same stripped form.

        onnx: ``"diactag-1.0"`` only. Load the int8 ONNX export — 3x faster and
              4x smaller on CPU for +0.03pp DER, with language auto-detection
              intact. Requires ``olaverse[onnx]``.

        device: ``"diactag-1.0"`` only. ``"cpu"`` (default), ``"cuda"`` or
              ``"mps"``. Ignored when ``onnx=True``.
    """

    def __init__(self, model: str = "diacnet-yor-viterbi", lang: str = None,
                 split_sentences: bool = True, splitter: "callable" = None,
                 min_confidence: float = 0.0, use_lexicon: bool = False,
                 onnx: bool = False, device: str = "cpu"):
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
            )
            # Fail on an unsupported code now rather than silently
            # auto-detecting on the first restore() call.
            self.diactag_lang = self.neural_decoder.normalize_language(lang)

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
        Identify the language of ``text``. ``"diactag-1.0"`` only — it is the
        only model with a language-identification head of its own.

        Returns:
            tuple: ``(iso_639_3_code, probability)``.
        """
        if self.method != "diactag":
            raise ValueError(
                f"detect_language() is only available on 'diactag-1.0'; this "
                f"Diacritizer is using '{self.method}'. For standalone language "
                f"identification use olaverse.nlp.LIDLite5 / LIDNeural25."
            )
        return self.neural_decoder.detect_language(text)

    def restore(self, text: str, lang: str = None, min_confidence: float = None,
                return_details: bool = False) -> Union[str, Tuple[str, List]]:
        """
        Restore diacritics in the given text.

        Args:
            text: Plain text (tones/diacritics stripped or missing).
            lang: Per-call language override for the multilingual models,
                  replacing the one given at construction.
            min_confidence: ``"diactag-1.0"`` only. Per-call abstention
                  threshold, so one loaded model can serve a CMS pre-fill and a
                  legal pipeline at different points on the coverage curve.
            return_details: ``"diactag-1.0"`` only. Also return per-character
                  results (``char``, ``confidence``, ``abstained``,
                  ``protected``) for routing low-confidence spans to review.

        Returns:
            Union[str, Tuple[str, List]]: the restored text, or ``(text, details)`` when ``return_details=True``.
        """
        if return_details and self.method != "diactag":
            raise ValueError(
                f"return_details=True is only supported by 'diactag-1.0', which "
                f"scores each character independently; this Diacritizer is using "
                f"'{self.method}'."
            )
        if min_confidence is not None and self.method != "diactag":
            raise ValueError(
                f"min_confidence is only supported by 'diactag-1.0'; this "
                f"Diacritizer is using '{self.method}'."
            )

        if self.method == "diactag":
            return self.neural_decoder.decode(
                text,
                lang=lang if lang is not None else self.diactag_lang,
                min_confidence=min_confidence,
                return_details=return_details,
            )

        if self.method == "auto":
            return self._auto_restore(text)

        if self.method == "diacnet":
            return self.neural_decoder.decode(
                text,
                lang=lang or self.diacnet_lang,
                split_sentences=self.split_sentences,
                splitter=self.splitter,
            )

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
