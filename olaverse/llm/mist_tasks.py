"""
Task-specific MIST models — small, single-purpose generators.

Unlike the general-purpose `MIST` chat family, these are fine-tuned for one job
each and expect an exact input format. Both wrappers bake that format in, so
callers pass plain text and get plain results back.

Requires: pip install olaverse[deeplearning]
"""
from __future__ import annotations

import json
import re
import warnings

_TITLE_REPOS = {
    "0.3b": "olaverse/mist-tg-0.3b",
}

_QUESTION_REPOS = {
    "1.5b": "olaverse/mist-qg-1.5b",
}

# The 25 languages mist-qg-1.5b was trained on, keyed by ISO 639-3 (the codes the
# reference demo uses). Values are the English names the model saw at training time —
# the prompt interpolates the name, not the code.
QG_LANGUAGES = {
    "eng": "English",     "fra": "French",      "deu": "German",
    "spa": "Spanish",     "por": "Portuguese",  "ita": "Italian",
    "nld": "Dutch",       "rus": "Russian",     "pol": "Polish",
    "tur": "Turkish",     "vie": "Vietnamese",  "ind": "Indonesian",
    "hin": "Hindi",       "jpn": "Japanese",    "kor": "Korean",
    "yor": "Yoruba",      "ibo": "Igbo",        "hau": "Hausa",
    "swh": "Swahili",     "amh": "Amharic",     "zul": "Zulu",
    "xho": "Xhosa",       "sna": "Shona",       "som": "Somali",
    "afr": "Afrikaans",
}

# ISO 639-1 aliases, so both qg.generate(..., language="yo") and "yor" work.
_QG_ISO1_ALIASES = {
    "en": "eng", "fr": "fra", "de": "deu", "es": "spa", "pt": "por",
    "it": "ita", "nl": "nld", "ru": "rus", "pl": "pol", "tr": "tur",
    "vi": "vie", "id": "ind", "hi": "hin", "ja": "jpn", "ko": "kor",
    "yo": "yor", "ig": "ibo", "ha": "hau", "sw": "swh", "am": "amh",
    "zu": "zul", "xh": "xho", "sn": "sna", "so": "som", "af": "afr",
}

# Flagged on the model card as lower-confidence. Not blocked — callers can still
# use them, but generate() warns so poor output isn't mistaken for a bug.
QG_WEAK_LANGUAGES = {"amh", "som", "sna"}

# Passages shorter than this give the model too little to work with.
_QG_MIN_PASSAGE_CHARS = 20

_QG_SYSTEM = "You write search-style questions that a passage directly answers."

# The teacher prompt used to distill this model from Qwen2.5-32B-Instruct — the
# wording the model was actually trained against. Note {slots}: the JSON skeleton
# carries exactly n placeholders, which is what keeps the model returning n items.
_QG_USER_TEMPLATE = """You are given a passage. Write {n} questions that the passage directly answers.
Rules:
- Every question MUST be answerable using ONLY this passage.
- NEVER copy or repeat a sentence from the passage.
- Rewrite the information into a natural question.
- Questions should sound like something a real person would ask in a search engine.
- Do not quote the passage.
- Vary the question types: factual, yes/no, why/how, comparison.
- Write all questions in {language}.
- Return ONLY valid JSON:
{{"questions": [{slots}]}}
Passage: {passage}"""


def _require_transformers(cls_name: str):
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise ImportError(
            f"transformers and torch are required to load {cls_name}. "
            "Install with: pip install olaverse[deeplearning]"
        )


def _pick_device(device: str | None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MISTTitleGenerator:
    """
    Short chat titles from a user's first message.

    Wraps olaverse/mist-tg-0.3b — a byte-level seq2seq model fine-tuned on real
    English chat messages and their titles. No prompt template or language tag
    is needed; the raw message goes straight in.

    Requires: pip install olaverse[deeplearning]

    Quick start:
        >>> titler = MISTTitleGenerator()
        >>> titler.generate("My laptop freezes whenever I open too many tabs, why?")
        'Laptop Freezing Impact'

    Batch:
        >>> titler.generate_batch(["How do I center a div?", "Best jollof recipe?"])

    Language support:
        Trained on English only. Latin-script languages often work as a
        byte-level transfer side effect, but are not guaranteed. Non-Latin
        scripts (CJK, Hangul, Devanagari, Ethiopic) are not supported — the
        model emits unrelated English text rather than a same-language title.
    """

    def __init__(self, size: str = "0.3b", device: str = None):
        """
        Args:
            size: Model variant. Currently only "0.3b". Also accepts a full
                  Hugging Face model ID.
            device: Torch device string ("cuda", "mps", "cpu"). Auto-detected
                    when omitted.
        """
        size_lower = size.lower()
        self.model_name = _TITLE_REPOS.get(size_lower, size)
        self.size = size_lower
        self._device_arg = device
        self.device = None
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """Download and load the model (runs once; cached after first call)."""
        if self._loaded:
            return

        _require_transformers("MISTTitleGenerator")
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.device = _pick_device(self._device_arg)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True

    def generate(self, message: str, max_new_tokens: int = 32) -> str:
        """
        Generate a short title for a chat message.

        Args:
            message: The user's message. Truncated to 256 bytes — the model's
                     trained input length.
            max_new_tokens: Maximum title length in tokens (bytes).

        Returns:
            str: the generated title.
        """
        return self.generate_batch([message], max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, messages: list, max_new_tokens: int = 32) -> list:
        """
        Generate titles for several messages in one forward pass.

        Args:
            messages: List of message strings.
            max_new_tokens: Maximum title length in tokens (bytes).

        Returns:
            list[str]: one title per input message, in order.
        """
        if not self._loaded:
            self.load()

        import torch

        inputs = self._tokenizer(
            list(messages),
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)

        return [t.strip() for t in self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)]


class MISTQuestionGenerator:
    """
    Search-style questions generated from a passage, across 25 languages.

    Wraps olaverse/mist-qg-1.5b — a decoder-only model fine-tuned to emit strict
    JSON. Useful both as a question-generation endpoint and as a data factory for
    minting (query, positive) pairs to train retrievers and rerankers.

    Requires: pip install olaverse[deeplearning]

    This is same-language question generation: ``language`` names the language
    the passage is written in, and questions come back in that language. It is
    not a translation step — pointing it at an English passage and asking for
    Yoruba does not produce usable Yoruba.

    Quick start:
        >>> qg = MISTQuestionGenerator()
        >>> qg.generate("Tides are caused by the gravitational pull of the moon...")
        ['What causes ocean tides?', 'Does the sun affect tides?', ...]

    A passage in another language — pass an ISO code (639-3 or 639-1) or the
    English name of the language:
        >>> qg.generate(yoruba_passage, n=3, language="yor")
        >>> qg.generate(yoruba_passage, n=3, language="yo")
        >>> qg.generate(yoruba_passage, n=3, language="Yoruba")

    Supported languages:
        eng, fra, deu, spa, por, ita, nld, rus, pol, tur, vie, ind, hin,
        jpn, kor, yor, ibo, hau, swh, amh, zul, xho, sna, som, afr
    """

    def __init__(self, size: str = "1.5b", device: str = None):
        """
        Args:
            size: Model variant. Currently only "1.5b". Also accepts a full
                  Hugging Face model ID.
            device: Torch device string ("cuda", "mps", "cpu"). Auto-detected
                    when omitted.
        """
        size_lower = size.lower()
        self.model_name = _QUESTION_REPOS.get(size_lower, size)
        self.size = size_lower
        self._device_arg = device
        self.device = None
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """Download and load the model (runs once; cached after first call)."""
        if self._loaded:
            return

        _require_transformers("MISTQuestionGenerator")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = _pick_device(self._device_arg)
        # bfloat16 only on CUDA, matching the reference demo. Verified identical
        # greedy output to float32, so there is nothing to gain by forcing it
        # onto MPS/CPU where support is patchier.
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True

    @staticmethod
    def _resolve_language(language: str) -> str:
        """Accept ISO 639-3 ('yor'), ISO 639-1 ('yo'), or an English name ('Yoruba')."""
        key = language.strip().lower()
        code = _QG_ISO1_ALIASES.get(key, key)
        if code in QG_LANGUAGES:
            return code, QG_LANGUAGES[code]
        for c, name in QG_LANGUAGES.items():
            if name.lower() == key:
                return c, name
        raise ValueError(
            f"Unsupported language {language!r}. Use one of {sorted(QG_LANGUAGES)}, "
            f"an ISO 639-1 code, or the English name of that language."
        )

    @staticmethod
    def _parse_questions(text: str, n: int) -> list:
        """Pull the questions list out of the model's JSON, tolerating stray text."""
        try:
            payload = json.loads(text[text.index("{"):text.rindex("}") + 1])
            questions = payload["questions"]
            if isinstance(questions, list):
                return [str(q).strip() for q in questions if str(q).strip()][:n]
        except (ValueError, KeyError, TypeError):
            pass

        # Fallback: the model dropped or truncated its JSON — recover quoted strings.
        candidates = re.findall(r'"([^"\\]{4,})"', text)
        return [c.strip() for c in candidates if c.strip().lower() != "questions"][:n]

    def generate(self, passage: str, n: int = 3, language: str = "English", max_new_tokens: int = 250) -> list:
        """
        Generate questions that the passage directly answers.

        Args:
            passage: Source text the questions must be answerable from.
            n: How many questions to request.
            language: The language the passage is written in — an ISO 639-3
                      code ("yor"), an ISO 639-1 code ("yo"), or the English
                      name ("Yoruba"). Questions come back in this language;
                      it does not translate across languages.
            max_new_tokens: Generation budget. Raise it for large ``n``.

        Returns:
            list[str]: up to ``n`` questions. May be shorter if the model
            returned fewer, or if generation was cut off by max_new_tokens.
        """
        if not self._loaded:
            self.load()

        import torch

        passage = (passage or "").strip()
        if not passage:
            raise ValueError("passage is empty — nothing to generate questions from.")
        if len(passage) < _QG_MIN_PASSAGE_CHARS:
            warnings.warn(
                f"Passage is only {len(passage)} characters; the model was trained on "
                "paragraph-length input and short passages give poor questions.",
                stacklevel=2,
            )

        code, lang_name = self._resolve_language(language)
        if code in QG_WEAK_LANGUAGES:
            warnings.warn(
                f"{lang_name} is one of this model's lower-confidence languages — "
                "check the model card's benchmark numbers before relying on the output.",
                stacklevel=2,
            )

        messages = [
            {"role": "system", "content": _QG_SYSTEM},
            {"role": "user", "content": _QG_USER_TEMPLATE.format(
                n=n,
                language=lang_name,
                passage=passage,
                # One placeholder per requested question — this is what holds the
                # model to n items rather than defaulting to three.
                slots=", ".join(['"..."'] * int(n)),
            )},
        ]

        encoded = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        # transformers >= 4.57 returns a BatchEncoding here; older versions a bare tensor.
        if torch.is_tensor(encoded):
            encoded = {"input_ids": encoded, "attention_mask": torch.ones_like(encoded)}
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        pad_id = self._tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self._tokenizer.eos_token_id

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )

        text = self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return self._parse_questions(text, n)

    def generate_batch(self, passages: list, n: int = 3, language: str = "English", max_new_tokens: int = 250) -> list:
        """
        Generate questions for several passages.

        Args:
            passages: List of source texts, all in the same language.
            n: Questions per passage.
            language: The language the passages are written in.
            max_new_tokens: Generation budget per passage.

        Returns:
            list[list[str]]: one question list per passage, in order.
        """
        return [
            self.generate(p, n=n, language=language, max_new_tokens=max_new_tokens)
            for p in passages
        ]
