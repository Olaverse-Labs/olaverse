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

_TITLE_REPOS = {
    "0.3b": "olaverse/mist-tg-0.3b",
}

_QUESTION_REPOS = {
    "1.5b": "olaverse/mist-qg-1.5b",
}

# The 25 languages mist-qg-1.5b was trained on. Values are the English language
# names the model saw during training — the prompt interpolates the name, not the code.
QG_LANGUAGES = {
    "en": "English",     "fr": "French",      "de": "German",
    "es": "Spanish",     "pt": "Portuguese",  "it": "Italian",
    "nl": "Dutch",       "ru": "Russian",     "pl": "Polish",
    "tr": "Turkish",     "vi": "Vietnamese",  "id": "Indonesian",
    "hi": "Hindi",       "ja": "Japanese",    "ko": "Korean",
    "yo": "Yoruba",      "ig": "Igbo",        "ha": "Hausa",
    "sw": "Swahili",     "am": "Amharic",     "zu": "Zulu",
    "xh": "Xhosa",       "sn": "Shona",       "so": "Somali",
    "af": "Afrikaans",
}

_QG_SYSTEM = "You write search-style questions that a passage directly answers."

# Verbatim from the model card — the model was fine-tuned on this exact wording,
# so changes here degrade output quality and JSON adherence.
_QG_USER_TEMPLATE = """You are given a passage. Write {n} questions that the passage directly answers.

Rules:
- Each question must be answerable using ONLY this passage.
- Vary the type: factual, yes/no, and a comparison or "why/how".
- Natural, like a real user search query. Do NOT write "according to the passage".
- Write the questions in {language}.

Return ONLY JSON: {{"questions": ["...", "...", "..."]}}

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

    Quick start:
        >>> qg = MISTQuestionGenerator()
        >>> qg.generate("Tides are caused by the gravitational pull of the moon...")
        ['What causes ocean tides?', 'Does the sun affect tides?', ...]

    Another language — pass a code or an English language name:
        >>> qg.generate(passage, n=3, language="yo")
        >>> qg.generate(passage, n=3, language="Yoruba")

    Supported languages:
        en, fr, de, es, pt, it, nl, ru, pl, tr, vi, id, hi, ja, ko,
        yo, ig, ha, sw, am, zu, xh, sn, so, af
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
        dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True

    @staticmethod
    def _resolve_language(language: str) -> str:
        """Accept either an ISO code ('yo') or an English name ('Yoruba')."""
        key = language.strip().lower()
        if key in QG_LANGUAGES:
            return QG_LANGUAGES[key]
        for name in QG_LANGUAGES.values():
            if name.lower() == key:
                return name
        raise ValueError(
            f"Unsupported language {language!r}. Use one of "
            f"{sorted(QG_LANGUAGES)} or the English name of that language."
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

    def generate(self, passage: str, n: int = 3, language: str = "English", max_new_tokens: int = 200) -> list:
        """
        Generate questions that the passage directly answers.

        Args:
            passage: Source text the questions must be answerable from.
            n: How many questions to request.
            language: Output language — an ISO code ("yo") or English name
                      ("Yoruba"). Must be one of the 25 supported languages.
            max_new_tokens: Generation budget. Raise it for large ``n``.

        Returns:
            list[str]: up to ``n`` questions. May be shorter if the model
            returned fewer, or if generation was cut off by max_new_tokens.
        """
        if not self._loaded:
            self.load()

        import torch

        lang_name = self._resolve_language(language)
        messages = [
            {"role": "system", "content": _QG_SYSTEM},
            {"role": "user", "content": _QG_USER_TEMPLATE.format(
                n=n, language=lang_name, passage=passage
            )},
        ]

        encoded = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        # transformers >= 4.57 returns a BatchEncoding here; older versions a bare tensor.
        if not torch.is_tensor(encoded):
            encoded = encoded["input_ids"]
        input_ids = encoded.to(self.device)

        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )

        text = self._tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        return self._parse_questions(text, n)

    def generate_batch(self, passages: list, n: int = 3, language: str = "English", max_new_tokens: int = 200) -> list:
        """
        Generate questions for several passages.

        Args:
            passages: List of source texts.
            n: Questions per passage.
            language: Output language, applied to every passage.
            max_new_tokens: Generation budget per passage.

        Returns:
            list[list[str]]: one question list per passage, in order.
        """
        return [
            self.generate(p, n=n, language=language, max_new_tokens=max_new_tokens)
            for p in passages
        ]
