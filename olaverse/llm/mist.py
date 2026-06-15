import os

_ALIASES = {
    "8b":       "olaverse/MIST-Mini-8B",
    "mini":     "olaverse/MIST-Mini-8B",
    "70b":      "olaverse/MIST-1-70B",
    "140b":     "olaverse/MIST-1-140B",
    "140b-4bit":"olaverse/MIST-1-140B-4bit",
    "thinking": "olaverse/MIST-Mini-8B-Thinking",
}

# EOS tokens differ per variant — critical for clean output.
# 8B/Thinking: ChatML <|im_end|> (128040) survived the DARE+TIES merge alongside Llama 3.1 tokens.
# 70B/140B: pure Llama 3.1 lineage — no ChatML token, use <|eom_id|> (128008) instead.
_EOS_TOKENS = {
    "olaverse/MIST-Mini-8B":        [128040, 128009, 128001],
    "olaverse/MIST-1-70B":          [128009, 128001, 128008],
    "olaverse/MIST-1-140B":         [128009, 128001, 128008],
    "olaverse/MIST-1-140B-4bit":    [128009, 128001, 128008],
    "olaverse/MIST-Mini-8B-Thinking":[128040, 128009, 128001],
}

# Verified generation defaults from model cards — models ramble without repetition_penalty + min_p.
_DEFAULTS = {
    "olaverse/MIST-Mini-8B":        {"temperature": 0.7, "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.5},
    "olaverse/MIST-1-70B":          {"temperature": 0.8, "top_p": 0.99, "min_p": 0.05, "repetition_penalty": 1.0},
    "olaverse/MIST-1-140B":         {"temperature": 0.7, "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.5},
    "olaverse/MIST-1-140B-4bit":    {"temperature": 0.7, "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.5},
    "olaverse/MIST-Mini-8B-Thinking":{"temperature": 0.6, "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.5},
}

_SYSTEM_PROMPTS = {
    "olaverse/MIST-Mini-8B":
        "You are MIST, a highly capable AI assistant. Be concise and helpful.",
    "olaverse/MIST-1-70B":
        "You are MIST, a highly capable AI assistant. Be concise and helpful.",
    "olaverse/MIST-1-140B":
        "You are MIST, a highly capable AI assistant. Be concise and helpful.",
    "olaverse/MIST-1-140B-4bit":
        "You are MIST, a highly capable AI assistant. Be concise and helpful.",
    "olaverse/MIST-Mini-8B-Thinking":
        "You are MIST, a highly capable reasoning AI. Before answering, always write your "
        "full reasoning inside <think> and </think> tags, then give your final answer after.",
}


class MIST:
    """
    Unified interface for the MIST model family by olaverse.

    Handles correct stop tokens, verified sampling defaults, and a local/hosted endpoint
    switch — all things a bare from_pretrained call gets wrong.

    Models (size=):
        "8b" / "mini"  — MIST-Mini-8B    (8B,   ~63 tok/s, fast everyday use)
        "70b"          — MIST-1-70B      (70B,  ~23 tok/s, structured, detailed)
        "140b"         — MIST-1-140B     (140B, ~8 tok/s,  deepest reasoning)
        "140b-4bit"    — MIST-1-140B-4bit (140B quantized, single H100/H200)
        "thinking"     — MIST-Mini-8B-Thinking (8B reasoning, shows <think> steps)

    Endpoints (endpoint=):
        "local"        — transformers local inference  (pip install olaverse[deeplearning])
        "featherless"  — Featherless.ai hosted API     (pip install olaverse[hosted])
        Any URL        — OpenAI-compatible endpoint    (Modal/vLLM, etc.)

    Quick start — local:
        >>> model = MIST(size="8b")
        >>> model.load()
        >>> print(model.generate("Explain DARE+TIES merging in one paragraph."))

    Quick start — hosted:
        >>> model = MIST(size="70b", endpoint="featherless", api_key="your-key")
        >>> print(model.generate("Write a Python retry decorator."))

    Multi-turn chat:
        >>> messages = [
        ...     {"role": "user", "content": "What is MIST?"},
        ...     {"role": "assistant", "content": "MIST is a merged model family..."},
        ...     {"role": "user", "content": "How large is the 140B version?"},
        ... ]
        >>> print(model.chat(messages))
    """

    def __init__(
        self,
        size: str = "8b",
        endpoint: str = "local",
        api_key: str = None,
        quantize: bool = False,
        system_prompt: str = None,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        Args:
            size: Model variant. One of "8b", "mini", "70b", "140b", "140b-4bit", "thinking".
                  Also accepts a full Hugging Face model ID.
            endpoint: "local", "featherless", or a custom base URL (e.g. your Modal deployment).
            api_key: API key for hosted endpoints. Falls back to FEATHERLESS_API_KEY env var.
            quantize: If True and endpoint="local", loads in 4-bit NF4 (requires bitsandbytes).
            system_prompt: Override the default system prompt for all calls.
            max_retries: Number of retry attempts on capacity/server errors (hosted only).
                         Set to 1 to disable retries. Defaults to 3.
            retry_delay: Base delay in seconds between retries. Each attempt waits
                         ``retry_delay * attempt`` seconds. Defaults to 5.0.
        """
        size_lower = size.lower()
        self.model_name = _ALIASES.get(size_lower, size)

        self.endpoint = endpoint
        self.api_key = api_key
        self.quantize = quantize
        self.system_prompt = system_prompt or _SYSTEM_PROMPTS.get(
            self.model_name, "You are MIST, a highly capable AI assistant. Be concise and helpful."
        )
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = float(retry_delay)

        self._defaults = _DEFAULTS.get(
            self.model_name,
            {"temperature": 0.7, "top_p": 0.95, "min_p": 0.05, "repetition_penalty": 1.5},
        )
        self._eos_tokens = _EOS_TOKENS.get(self.model_name, [128009, 128001])

        self._model = None
        self._tokenizer = None
        self._client = None
        self._loaded = False

    def load(self):
        """
        Load the model. Required before generate()/chat() when endpoint='local'.
        For hosted endpoints this initialises the API client instead.
        Safe to call multiple times — no-op after the first load.
        """
        if self._loaded:
            return

        if self.endpoint != "local":
            self._init_client()
            self._loaded = True
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch are required for local inference. "
                "Install with: pip install olaverse[deeplearning]"
            )

        load_kwargs = {"torch_dtype": "auto", "device_map": "auto"}

        if self.quantize:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for quantized loading. "
                    "Install with: pip install bitsandbytes"
                )
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self._model.eval()
        self._loaded = True

    def _init_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' library is required for hosted inference. "
                "Install with: pip install olaverse[hosted]"
            )

        if self.endpoint == "featherless":
            base_url = "https://api.featherless.ai/v1"
            key = self.api_key or os.environ.get("FEATHERLESS_API_KEY", "token")
        else:
            # Custom URL (Modal, vLLM, etc.) — append /v1 if not already present
            base_url = self.endpoint if "/v1" in self.endpoint else self.endpoint.rstrip("/") + "/v1"
            key = self.api_key or "token"

        self._client = OpenAI(api_key=key, base_url=base_url)

    def _build_messages(self, prompt_or_messages, system=None):
        """Normalise a string or messages list into the OpenAI messages format."""
        sys = system or self.system_prompt
        if isinstance(prompt_or_messages, str):
            return [
                {"role": "system", "content": sys},
                {"role": "user",   "content": prompt_or_messages},
            ]
        msgs = list(prompt_or_messages)
        if not msgs or msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": sys}] + msgs
        return msgs

    def generate(self, prompt: str, system: str = None, max_new_tokens: int = 1024, stream: bool = False, **kwargs: float) -> str:
        """
        Single-turn generation from a plain string prompt.

        Args:
            prompt: User message.
            system: Per-call system prompt override.
            max_new_tokens: Maximum tokens to generate.
            stream: Return a generator of partial strings instead of a full string.
                    Only supported for hosted endpoints.
            **kwargs: Override any default generation param
                      (temperature, top_p, min_p, repetition_penalty).

        Returns:
            str, or generator[str] when stream=True.
        """
        return self.chat(
            self._build_messages(prompt, system),
            max_new_tokens=max_new_tokens,
            stream=stream,
            **kwargs,
        )

    def chat(self, messages: list, max_new_tokens: int = 1024, stream: bool = False, **kwargs: float) -> str:
        """
        Multi-turn generation from a messages list.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
                      A system message is prepended automatically if not present.
            max_new_tokens: Maximum tokens to generate.
            stream: Return a generator of partial strings (hosted endpoints only).
            **kwargs: Override generation parameters.

        Returns:
            str, or generator[str] when stream=True.
        """
        if not self._loaded:
            self.load()

        params = {**self._defaults, **kwargs}

        if self.endpoint == "local":
            return self._generate_local(messages, max_new_tokens, params)
        return self._generate_hosted(messages, max_new_tokens, params, stream=stream)

    def _generate_local(self, messages, max_new_tokens, params):
        import torch

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            inputs = inputs.to("mps")

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "repetition_penalty": params["repetition_penalty"],
            "eos_token_id": self._eos_tokens,
            "pad_token_id": 128001,
        }
        # min_p requires transformers >= 4.45; pass it through and let transformers ignore it gracefully
        if "min_p" in params:
            gen_kwargs["min_p"] = params["min_p"]

        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _generate_hosted(self, messages, max_new_tokens, params, stream=False):
        import time
        from openai import APIError

        if self._client is None:
            self._init_client()

        rep_penalty = params.get("repetition_penalty", 1.0)

        completion_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": float(params["temperature"]),
            "top_p": float(params["top_p"]),
            # OpenAI API uses frequency_penalty (0–2); map from repetition_penalty (1–2) → (0–1)
            "frequency_penalty": max(0.0, float(rep_penalty) - 1.0),
            "stream": stream,
        }

        if stream:
            return self._stream_chunks(completion_kwargs)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**completion_kwargs)
                return response.choices[0].message.content
            except APIError as e:
                last_error = e
                is_capacity = any(kw in str(e).lower() for kw in ("capacity", "overloaded", "529", "503"))
                if is_capacity and attempt < self.max_retries - 1:
                    wait = self.retry_delay * (attempt + 1)
                    time.sleep(wait)
                else:
                    break

        raise last_error

    def _stream_chunks(self, completion_kwargs):
        for chunk in self._client.chat.completions.create(**completion_kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
