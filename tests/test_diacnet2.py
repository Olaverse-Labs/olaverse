"""
Tests for diacnet-2.0 and diacnet-mini-2.0 (ByT5 text-to-text diacritizers).

Tiers:

  * pure — output alignment, chunking, hint and tag helpers. No torch, no
    network. The alignment tests include the invariant the whole design leans on:
    aligned output strips back to the input, whatever the model produced.
  * mocked — the decoder and the ``Diacritizer`` wrapper against a fake
    tokenizer and model: prompt format, chunking, greedy generation arguments,
    length-sorted padded batching, dtype/device, validation. Nothing is
    downloaded.
  * ``slow`` — the real private checkpoints and the golden examples. Excluded by
    default (``pytest -m slow``) and skipped when no Hugging Face token is
    available.

The golden strings below came from GPU bf16 batched runs. If one differs on a
CPU float32 run by a single mark, that is a result to report, not a reason to
loosen the assertion.
"""

import random
import sys
import types
import unicodedata

import pytest

from olaverse.nlp import diacritizer as dz
from olaverse.nlp.diacnet_utils import (
    DIACNET2_TAGS,
    align,
    build_hint,
    chunk_text,
    resolve_tag,
    strip_marks,
)
from olaverse.nlp.diacritizer import MODEL_REGISTRY, DiacNet2Decoder, Diacritizer

slow = pytest.mark.slow

GOLDEN = [
    ("yor", "", "O so fun ara re pe oun ko ni isoro kankan.",
     "Ó sọ fún ara rẹ̀ pé òun kò ní ìṣòro kankan."),
    ("hau", "", "Cutar kan dauki tsawon kwana 14 zuwa 21 kafin ya warke.",
     "Cutar kan ɗauki tsawon kwana 14 zuwa 21 kafin ya warke."),
    ("pol", "", "Facebook zawiesil jedno z moich szesciu kont.",
     "Facebook zawiesił jedno z moich sześciu kont."),
    ("ara", "", "وهذا قول مرغوب عنه .",
     "وَهَذَا قَوْلٌ مَرْغُوبٌ عَنْهُ ."),
    ("vie", "[g: ranh=free (time)]",
     "Chi ay chi that su ranh vao nhung buoi toi sau khi da cho con ngu say.",
     "Chị ấy chỉ thật sự rảnh vào những buổi tối sau khi đã cho con ngủ say."),
]


def _hf_token():
    pytest.importorskip("transformers")
    from huggingface_hub import get_token
    return get_token()


def _needs_token():
    if not _hf_token():
        pytest.skip("needs a Hugging Face token (the olaverse/* repos are private)")


# =========================================================================== #
# Alignment — pure
# =========================================================================== #

def test_align_drops_an_added_character():
    assert align("se eranko naa si gbo o", "ṣe ẹranko náà sì gbọ́ ọ?") == \
        "ṣe ẹranko náà sì gbọ́ ọ"


def test_align_returns_marked_output_when_only_marks_were_added():
    assert align("o so fun ara re", "ó sọ fún ara rẹ") == "ó sọ fún ara rẹ"


def test_align_falls_back_to_the_input_letter_when_a_letter_changed():
    # the model turned the final 'a' into 'u': keep the user's 'a', take the mark on 'o'
    assert align("ola", "ọlu") == "ọla"
    # a swapped / typo-fixed letter never reaches the output
    assert align("recieve", "receive") == "recieve"


def test_align_never_adds_or_removes_a_letter():
    assert align("abc", "abcd") == "abc"
    assert align("abcd", "abc") == "abcd"
    assert align("abc", "xyz") == "abc"
    assert align("", "xyz") == ""


def test_align_keeps_hausa_hooks():
    out = align("kasar Hausa bara dan yaya", "ƙasar Hausa ɓara ɗan ƴaya")
    assert out == "ƙasar Hausa ɓara ɗan ƴaya"
    assert strip_marks(out) == "kasar Hausa bara dan yaya"


def test_align_keeps_other_barred_and_dotless_letters():
    assert align("ldz dung", "łdz đung") == "łdz đung"
    assert align("kisa", "kısa") == "kısa"             # Turkish dotless i


def test_align_takes_the_marks_from_the_model_even_over_typed_ones():
    """Only *letters* are protected. A hook is a mark (ɗ and d share a base), so
    if the model answers with the plain letter, the model's choice is used —
    this is the reference function's behaviour. The letters are still the
    user's: the skeleton is identical either way."""
    out = align("ɗan", "dan")
    assert out == "dan"
    assert strip_marks(out) == strip_marks("ɗan")


def test_align_is_unicode_normalisation_insensitive():
    nfd = unicodedata.normalize("NFD", "sọ́ fún")
    nfc = unicodedata.normalize("NFC", "sọ́ fún")
    assert align("so fun", nfd) == align("so fun", nfc)


def test_align_arabic_never_adds_or_drops_a_hamza():
    # model dropped the hamza: input keeps it
    assert align("أحمد", "احمد") == "أحمد"
    # model invented one: input stays hamza-free
    assert align("احمد", "أحمد") == "احمد"
    # hamza/madda are spelling, not marks: a changed alef is a changed letter
    assert align("آمن", "امن") == "آمن"


def test_align_arabic_takes_the_harakat():
    assert align("وهذا قول مرغوب عنه .", "وَهَذَا قَوْلٌ مَرْغُوبٌ عَنْهُ .") == \
        "وَهَذَا قَوْلٌ مَرْغُوبٌ عَنْهُ ."
    # punctuation spacing the model changed does not leak into the text
    assert align("وهذا قول .", "وَهَذَا قَوْلٌ.") == "وَهَذَا قَوْلٌ ."


def test_align_keeps_marks_already_in_the_input_where_the_model_agrees():
    # the user typed the underdots; the model adds tones
    assert align("Ki lo ṣe ti inu rẹ", "Kí ló ṣe tí inú rẹ̀") == "Kí ló ṣe tí inú rẹ̀"


# --- the invariant ---------------------------------------------------------

_BARE = [
    "se eranko naa si gbo o",
    "Toi khong biet tieng Viet",
    "Zazolc gesla jazn, lodz i cma.",
    "Igdir'in cigir acan",
    "Cutar kan dauki tsawon kwana 14",
    "ذهب الطالب إلى المدرسة في الصباح",
    "أحمد يقرأ الكتاب وآمن المؤمنون وئد",
    "El nino comio manana, cuanto?",
    "Ou etes-vous ? Ca coute cher.",
]
_MARKED = [
    "ṣé ẹranko náà sì gbọ́ ọ?",
    "Tôi không biết tiếng Việt",
    "Zażółć gęślą jaźń, łódź i ćma.",
    "ƙasar Hausa ɓarna ɗan yaƴi ƙwarai",
    "ذَهَبَ الطَّالِبُ إِلَى الْمَدْرَسَةِ فِي الصَّبَاحِ",
    "أَحْمَدُ يَقْرَأُ الْكِتَابَ وَآمَنَ الْمُؤْمِنُونَ",
    "Où êtes-vous ? Ça coûte cher.",
    "Não é possível à mãe, coração.",
]
_NOISE = "abcdeoiuxzqاأبتمنهوي"


def _mutate(text, rng):
    """Corrupt a 'model output' the ways a generative model does."""
    chars = list(text)
    for _ in range(rng.randint(0, 6)):
        if not chars:
            break
        op = rng.choice(["drop", "replace", "insert", "swap", "mark", "dupe", "strip"])
        i = rng.randrange(len(chars))
        if op == "drop":
            del chars[i]
        elif op == "replace":
            chars[i] = rng.choice(_NOISE)
        elif op == "insert":
            chars.insert(i, rng.choice(_NOISE + "?.,! "))
        elif op == "swap" and i + 1 < len(chars):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "mark":
            chars.insert(i + 1, rng.choice("َُّ̣́̀ٔ"))
        elif op == "dupe":
            chars.insert(i, chars[i])
        elif op == "strip":
            chars[i] = strip_marks(chars[i]) or chars[i]
    return "".join(chars)


@pytest.mark.parametrize("bare", _BARE)
def test_aligned_output_strips_back_to_the_bare_input_exactly(bare):
    """For a bare input, stripping the marks of aligned output gives the input."""
    rng = random.Random(hash(bare) & 0xFFFF)
    fully_marked = {
        "se eranko naa si gbo o": "ṣé ẹranko náà sì gbọ́ ọ",
        "ذهب الطالب إلى المدرسة في الصباح": _MARKED[4],
    }.get(bare, bare)
    for _ in range(200):
        out = _mutate(fully_marked, rng)
        assert strip_marks(align(bare, out)) == bare


@pytest.mark.parametrize("src", _BARE + _MARKED)
def test_aligned_output_has_the_inputs_letters_whatever_the_model_did(src):
    rng = random.Random(len(src))
    donors = _MARKED + _BARE
    for _ in range(200):
        out = _mutate(rng.choice(donors), rng)
        assert strip_marks(align(src, out)) == strip_marks(src)


def test_aligned_output_is_exactly_the_input_when_the_model_returns_it_unchanged():
    for s in _BARE + _MARKED:
        assert align(s, s) == unicodedata.normalize("NFC", s)


def test_align_with_unrelated_or_empty_output_keeps_the_input():
    for s in _BARE:
        assert strip_marks(align(s, "")) == s
        assert align(s, "") == unicodedata.normalize("NFC", s)


# =========================================================================== #
# Chunking, hints, tags — pure
# =========================================================================== #

def test_chunks_are_at_most_the_limit_and_rejoin_with_spaces():
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)
    chunks = chunk_text(text, 300)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert " ".join(chunks) == text


def test_short_text_is_one_chunk_and_empty_is_none():
    assert chunk_text("o so fun ara re") == ["o so fun ara re"]
    assert chunk_text("") == []


def test_a_single_overlong_word_is_kept_whole():
    word = "x" * 500
    assert chunk_text(f"a {word} b", 300) == ["a", word, "b"]


def test_chunks_split_on_spaces_only():
    text = ("aaa\nbbb " * 100).strip()                  # a newline is not a boundary
    chunks = chunk_text(text, 100)
    assert len(chunks) > 1 and all(len(c) <= 100 for c in chunks)
    assert all(w == "aaa\nbbb" for c in chunks for w in c.split(" "))
    assert " ".join(chunks) == text


def test_build_hint_formats_and_validates():
    assert build_hint(None) == "" and build_hint("") == "" and build_hint({}) == ""
    assert build_hint({"ranh": "free (time)"}) == "[g: ranh=free (time)]"
    assert build_hint({"ogun": "twenty", "ranh": "free"}) == "[g: ogun=twenty | ranh=free]"
    assert build_hint(["ogun=twenty", "ranh=free"]) == "[g: ogun=twenty | ranh=free]"
    assert build_hint([("ogun", "twenty")]) == "[g: ogun=twenty]"
    assert build_hint("ogun=twenty | ranh=free") == "[g: ogun=twenty | ranh=free]"
    assert build_hint("[g: ranh=free]") == "[g: ranh=free]"        # ready-made, passed through
    for bad in ("noequals", {"a": ""}, ["a=b", "oops"], {"a": "b|c"}, {"a]": "b"}):
        with pytest.raises(ValueError):
            build_hint(bad)


@pytest.mark.parametrize("lang,tag", [
    ("yor", "yor"), ("yo", "yor"), ("YO", "yor"), ("ibo", "ibo"), ("ig", "ibo"),
    ("hau", "hau"), ("ha", "hau"), ("vie", "vie"), ("vi", "vie"), ("pol", "pol"),
    ("pl", "pol"), ("tur", "tur"), ("tr", "tur"), ("por", "por"), ("pt", "por"),
    ("spa", "spa"), ("es", "spa"), ("fra", "fra"), ("fr", "fra"), ("ita", "ita"),
    ("it", "ita"), ("ara", "ara"), ("ar", "ara"), ("ara-nocase", "ara-nocase"),
    ("ara_nocase", "ara-nocase"), ("auto", "auto"), ("<yor>", "yor"), (None, "auto"),
])
def test_resolve_tag(lang, tag):
    assert resolve_tag(lang) == tag


def test_arabic_without_case_endings_uses_the_nocase_tag():
    assert resolve_tag("ara", case_endings=False) == "ara-nocase"
    assert resolve_tag("ar", case_endings=False) == "ara-nocase"
    assert resolve_tag("ara-nocase", case_endings=True) == "ara-nocase"   # explicit wins
    assert resolve_tag("yor", case_endings=False) == "yor"                # no effect elsewhere


def test_resolve_tag_rejects_unsupported_languages():
    with pytest.raises(ValueError, match="Unsupported language 'de'"):
        resolve_tag("de")
    assert set(DIACNET2_TAGS) == {"yor", "ibo", "hau", "vie", "pol", "tur", "por",
                                  "spa", "fra", "ita", "ara", "ara-nocase", "auto"}


# =========================================================================== #
# Registry and validation — no model
# =========================================================================== #

def test_registry_has_the_diacnet_2_models_and_keeps_the_old_ones():
    assert MODEL_REGISTRY["diacnet-2.0"] == {"lang": "multi", "method": "diacnet2"}
    assert MODEL_REGISTRY["diacnet-mini-2.0"] == {"lang": "multi", "method": "diacnet2"}
    assert MODEL_REGISTRY["diacnet-1.0"] == {"lang": "multi", "method": "diacnet"}
    assert MODEL_REGISTRY["diacnet-1.1"] == {"lang": "multi", "method": "diacnet"}


def test_new_options_are_rejected_on_models_that_do_not_have_them():
    d = Diacritizer(model="diacnet-yor-viterbi")
    with pytest.raises(ValueError, match="aligned"):
        d.restore("se eranko", aligned=True)
    with pytest.raises(ValueError, match="hints"):
        d.restore("se eranko", hints={"ogun": "war"})
    with pytest.raises(ValueError, match="case_endings"):
        d.restore("se eranko", case_endings=False)
    with pytest.raises(ValueError, match="hints"):
        d.restore_batch(["se eranko"], hints=[{"ogun": "war"}])


def test_new_constructor_options_are_rejected_on_models_that_do_not_have_them():
    with pytest.raises(ValueError, match="aligned"):
        Diacritizer(model="diacnet-yor-viterbi", aligned=True)
    with pytest.raises(ValueError, match="case_endings"):
        Diacritizer(model="auto", case_endings=False)


def test_restore_batch_on_a_classic_model_restores_each_text():
    d = Diacritizer(model="diacnet-yor-viterbi")
    texts = ["Ojo lo si oja", "Ojo lo si oja lana"]
    assert d.restore_batch(texts) == [d.restore(t) for t in texts]


# =========================================================================== #
# The decoder against a fake tokenizer and model
# =========================================================================== #

class _Enc(dict):
    def to(self, device):
        self.device = device
        return self


class _FakeTokenizer:
    def __init__(self):
        self.calls = []                  # (prompts, padding) per batch
        self.responder = None

    def __call__(self, prompts, return_tensors=None, padding=None):
        import torch
        prompts = list(prompts)
        self.calls.append((prompts, padding))
        width = max(len(p.encode("utf-8")) for p in prompts) + 1   # bytes + EOS
        ids = torch.zeros(len(prompts), width, dtype=torch.long)
        return _Enc(input_ids=ids, attention_mask=torch.ones_like(ids))

    def batch_decode(self, out, skip_special_tokens=True):
        batch = self.calls[-1][0]
        return [self.responder(batch[i]) for i in out[:, 0].tolist()]


class _FakeModel:
    def __init__(self):
        self.generate_kwargs = []
        self.moved_to = None
        self.is_eval = False

    def generate(self, **kw):
        import torch
        self.generate_kwargs.append(kw)
        return torch.arange(kw["input_ids"].shape[0])[:, None]

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.is_eval = True
        return self


def _strip_prompt(prompt):
    """What a perfect model sees as its payload: the text after the tag/hint."""
    body = prompt.split("> ", 1)[1]
    if body.startswith("[g:"):
        body = body.split("] ", 1)[1]
    return body


@pytest.fixture
def fake_transformers(monkeypatch):
    pytest.importorskip("torch")
    tok, model = _FakeTokenizer(), _FakeModel()
    tok.responder = _strip_prompt           # echo the text: aligned output == input
    load_kwargs = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kw):
            load_kwargs["tokenizer"] = (name, kw)
            return tok

    class AutoModelForSeq2SeqLM:
        @staticmethod
        def from_pretrained(name, **kw):
            load_kwargs["model"] = (name, kw)
            return model

    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = AutoTokenizer
    mod.AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM
    monkeypatch.setitem(sys.modules, "transformers", mod)
    monkeypatch.setattr(dz, "_NEURAL_CACHE", {})
    return types.SimpleNamespace(tok=tok, model=model, load=load_kwargs)


def test_decoder_loads_with_auto_classes_in_eval_mode(fake_transformers):
    import torch
    DiacNet2Decoder("olaverse/diacnet-2.0", device="cpu")
    assert fake_transformers.load["tokenizer"][0] == "olaverse/diacnet-2.0"
    assert fake_transformers.load["model"][0] == "olaverse/diacnet-2.0"
    assert fake_transformers.load["model"][1]["dtype"] is torch.float32
    assert fake_transformers.model.is_eval and fake_transformers.model.moved_to == "cpu"


def test_decoder_uses_bf16_on_cuda_and_float32_elsewhere(fake_transformers):
    import torch
    DiacNet2Decoder("olaverse/diacnet-2.0", device="cuda")
    assert fake_transformers.load["model"][1]["dtype"] is torch.bfloat16
    assert fake_transformers.model.moved_to == "cuda"
    DiacNet2Decoder("olaverse/diacnet-2.0", device="cuda:1")
    assert fake_transformers.load["model"][1]["dtype"] is torch.bfloat16
    DiacNet2Decoder("olaverse/diacnet-2.0", device="cpu")
    assert fake_transformers.load["model"][1]["dtype"] is torch.float32
    DiacNet2Decoder("olaverse/diacnet-2.0", device="mps")
    assert fake_transformers.load["model"][1]["dtype"] is torch.float32


def test_prompt_format(fake_transformers):
    dec = DiacNet2Decoder(device="cpu")
    dec.decode("o so fun ara re", lang="yo")
    dec.decode("El nino esta en la casa")                       # no lang -> <auto>
    dec.decode("Chi ay chi that su ranh", lang="vie", hints={"ranh": "free (time)"})
    dec.decode("وهذا قول", lang="ar", case_endings=False)
    prompts = [c[0][0] for c in fake_transformers.tok.calls]
    assert prompts == [
        "<yor> o so fun ara re",
        "<auto> El nino esta en la casa",
        "<vie> [g: ranh=free (time)] Chi ay chi that su ranh",
        "<ara-nocase> وهذا قول",
    ]


def test_generation_is_greedy_with_a_token_budget_tied_to_the_input(fake_transformers):
    dec = DiacNet2Decoder(device="cpu")
    dec.decode("o so fun ara re", lang="yo")
    kw = fake_transformers.model.generate_kwargs[0]
    assert kw["num_beams"] == 1 and kw["do_sample"] is False
    assert kw["max_new_tokens"] == 2 * kw["input_ids"].shape[1] + 16


def test_long_text_is_chunked_at_300_characters_and_rejoined(fake_transformers):
    dec = DiacNet2Decoder(device="cpu", batch_size=4)
    text = " ".join(f"ogun{i}" for i in range(300))
    out = dec.decode(text, lang="yor")
    assert out == text
    prompts = [p for c in fake_transformers.tok.calls for p in c[0]]
    assert len(prompts) > 1
    assert all(len(_strip_prompt(p)) <= 300 for p in prompts)


def test_batches_are_padded_to_longest_sorted_by_length_and_order_is_restored(fake_transformers):
    dec = DiacNet2Decoder(device="cpu", batch_size=2)
    texts = ["a b", "ccccc dddd eeeeee", "ff", "gggggggggg hhhhhhhhhhh", "i"]
    out = dec.decode_batch(texts, lang="yor")
    assert out == texts                                       # order restored
    calls = fake_transformers.tok.calls
    assert all(padding == "longest" for _, padding in calls)
    assert [len(p) for p, _ in calls] == [2, 2, 1]            # batch_size honoured
    flat = [p for prompts, _ in calls for p in prompts]
    lengths = [len(p.encode("utf-8")) for p in flat]
    assert lengths == sorted(lengths, reverse=True)           # longest first


def test_per_call_batch_size_does_not_change_the_decoder(fake_transformers):
    dec = DiacNet2Decoder(device="cpu", batch_size=16)
    dec.decode_batch(["a", "b", "c"], lang="yor", batch_size=1)
    assert [len(p) for p, _ in fake_transformers.tok.calls] == [1, 1, 1]
    assert dec.batch_size == 16


def test_each_text_gets_its_own_language_and_hint(fake_transformers):
    dec = DiacNet2Decoder(device="cpu", batch_size=8)
    dec.decode_batch(["se eranko", "Toi khong", "x"], lang=["yor", "vi", None],
                     hints=[None, {"khong": "no"}, None])
    prompts = sorted(p for c in fake_transformers.tok.calls for p in c[0])
    assert prompts == sorted(["<yor> se eranko", "<vie> [g: khong=no] Toi khong", "<auto> x"])


def test_per_text_arguments_must_match_the_number_of_texts(fake_transformers):
    dec = DiacNet2Decoder(device="cpu")
    with pytest.raises(ValueError, match="lang has 1 entries for 2 texts"):
        dec.decode_batch(["a", "b"], lang=["yor"])
    with pytest.raises(ValueError, match="hints has 1 entries for 2 texts"):
        dec.decode_batch(["a", "b"], lang="yor", hints=[None])


def test_a_bare_string_is_not_silently_split_into_characters(fake_transformers):
    dec = DiacNet2Decoder(device="cpu")
    with pytest.raises(TypeError, match="not a single string"):
        dec.decode_batch("o so fun", lang="yor")
    with pytest.raises(TypeError, match="not a single string"):
        Diacritizer(model="diacnet-2.0").restore_batch("o so fun")
    with pytest.raises(TypeError, match="not a single string"):
        Diacritizer(model="diacnet-yor-viterbi").restore_batch("o so fun")
    assert fake_transformers.tok.calls == []


def test_empty_texts_come_back_empty_and_are_not_sent_to_the_model(fake_transformers):
    dec = DiacNet2Decoder(device="cpu")
    assert dec.decode("", lang="yor") == ""
    assert dec.decode("   ", lang="yor") == ""
    assert dec.decode_batch(["", "a b", None, "  "], lang="yor") == ["", "a b", "", ""]
    assert sum(len(p) for p, _ in fake_transformers.tok.calls) == 1
    assert dec.decode_batch([], lang="yor") == []


def test_aligned_is_the_default_and_keeps_the_input_letters(fake_transformers):
    fake_transformers.tok.responder = lambda p: "ṣe ẹranko náà sì gbọ́ ọ?"   # extra '?'
    dec = DiacNet2Decoder(device="cpu")
    assert dec.decode("se eranko naa si gbo o", lang="yor") == "ṣe ẹranko náà sì gbọ́ ọ"


def test_aligned_false_returns_the_raw_output_stripped_and_nfc(fake_transformers):
    decomposed = unicodedata.normalize("NFD", "  ṣé ẹranko náà sì gbọ́ ọ?  ")
    fake_transformers.tok.responder = lambda p: decomposed
    dec = DiacNet2Decoder(device="cpu")
    out = dec.decode("se eranko naa si gbo o", lang="yor", aligned=False)
    assert out == unicodedata.normalize("NFC", "ṣé ẹranko náà sì gbọ́ ọ?")
    assert out == out.strip()


def test_aligned_never_lets_a_changed_letter_through(fake_transformers):
    fake_transformers.tok.responder = lambda p: "sọ fún kẹ"       # 'ara' -> 'kẹ'
    dec = DiacNet2Decoder(device="cpu")
    out = dec.decode("so fun ara", lang="yor")
    assert strip_marks(out) == "so fun ara"


def test_chunk_outputs_are_aligned_one_by_one(fake_transformers):
    fake_transformers.tok.responder = lambda p: _strip_prompt(p).replace("e", "ẹ")
    dec = DiacNet2Decoder(device="cpu")
    text = " ".join(["ere"] * 200)
    assert dec.decode(text, lang="yor") == text.replace("e", "ẹ")


# =========================================================================== #
# The Diacritizer wrapper — diacnet-2.x
# =========================================================================== #

def test_wrapper_defaults_aligned_on_and_language_auto(fake_transformers):
    d = Diacritizer(model="diacnet-2.0")
    assert d.aligned is True
    fake_transformers.tok.responder = lambda p: "Ó sọ"
    assert d.restore("O so") == "Ó sọ"
    assert fake_transformers.tok.calls[-1][0] == ["<auto> O so"]


def test_wrapper_construction_options(fake_transformers):
    d = Diacritizer(model="diacnet-mini-2.0", lang="ar", case_endings=False,
                    device="cpu", batch_size=3)
    assert fake_transformers.load["model"][0] == "olaverse/diacnet-mini-2.0"
    d.restore("وهذا قول")
    assert fake_transformers.tok.calls[-1][0] == ["<ara-nocase> وهذا قول"]
    # per-call override of both language and case endings
    d.restore("وهذا قول", lang="ara", case_endings=True)
    assert fake_transformers.tok.calls[-1][0] == ["<ara> وهذا قول"]
    d.restore("o so", lang="yo")
    assert fake_transformers.tok.calls[-1][0] == ["<yor> o so"]


def test_wrapper_restore_with_hints(fake_transformers):
    d = Diacritizer(model="diacnet-2.0", lang="vie")
    d.restore("Chi ay ranh", hints={"ranh": "free (time)"})
    assert fake_transformers.tok.calls[-1][0] == ["<vie> [g: ranh=free (time)] Chi ay ranh"]
    d.restore("Chi ay ranh", hints=["ranh=free", "chi=only"])
    assert fake_transformers.tok.calls[-1][0] == ["<vie> [g: ranh=free | chi=only] Chi ay ranh"]


def test_wrapper_aligned_can_be_turned_off(fake_transformers):
    fake_transformers.tok.responder = lambda p: "ṣe ẹranko?"
    on = Diacritizer(model="diacnet-2.0", lang="yor")
    off = Diacritizer(model="diacnet-2.0", lang="yor", aligned=False)
    assert on.restore("se eranko") == "ṣe ẹranko"
    assert off.restore("se eranko") == "ṣe ẹranko?"
    assert on.restore("se eranko", aligned=False) == "ṣe ẹranko?"


def test_wrapper_restore_batch(fake_transformers):
    d = Diacritizer(model="diacnet-2.0", lang="yor", batch_size=2)
    texts = ["a b", "cc dd ee", "f"]
    assert d.restore_batch(texts) == texts
    assert [len(p) for p, _ in fake_transformers.tok.calls] == [2, 1]
    d.restore_batch(texts, lang=["yor", "ara", "auto"])
    prompts = sorted(p for c in fake_transformers.tok.calls[2:] for p in c[0])
    assert prompts == sorted(["<yor> a b", "<ara> cc dd ee", "<auto> f"])


def test_wrapper_validates_the_language_before_loading_anything(fake_transformers):
    with pytest.raises(ValueError, match="Unsupported language 'de'"):
        Diacritizer(model="diacnet-2.0", lang="de")
    assert "model" not in fake_transformers.load            # nothing was downloaded


def test_wrapper_rejects_diactag_only_options_on_diacnet_2(fake_transformers):
    d = Diacritizer(model="diacnet-2.0")
    with pytest.raises(ValueError, match="min_confidence"):
        d.restore("o so", min_confidence=0.9)
    with pytest.raises(ValueError, match="return_details"):
        d.restore("o so", return_details=True)
    with pytest.raises(ValueError, match="detect_language"):
        d.detect_language("o so")


def test_weights_are_shared_per_model_and_device(fake_transformers):
    a = Diacritizer(model="diacnet-2.0", device="cpu", batch_size=2)
    b = Diacritizer(model="diacnet-2.0", device="cpu", batch_size=8)
    assert a.neural_decoder is b.neural_decoder
    assert (a.batch_size, b.batch_size) == (2, 8)


# =========================================================================== #
# diacnet-1.x — must behave exactly as before
# =========================================================================== #

@pytest.fixture
def fake_diacnet_1(monkeypatch):
    pytest.importorskip("torch")
    import torch

    class Tok:
        def __init__(self):
            self.prompts = []

        def __call__(self, text, return_tensors=None, truncation=None, max_length=None):
            self.prompts.append(text)
            return {"input_ids": torch.zeros(1, 4, dtype=torch.long)}

        def decode(self, ids, skip_special_tokens=True):
            return self.reply

    class Model:
        def eval(self):
            return self

        def generate(self, **kw):
            return torch.zeros(1, 2, dtype=torch.long)

    tok = Tok()
    tok.reply = "Ó sọ fún?"
    mod = types.ModuleType("transformers")
    mod.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda n, **k: tok)
    mod.T5ForConditionalGeneration = types.SimpleNamespace(from_pretrained=lambda n, **k: Model())
    monkeypatch.setitem(sys.modules, "transformers", mod)
    monkeypatch.setattr(dz, "_NEURAL_CACHE", {})
    return tok


def test_diacnet_1_1_output_is_unchanged_by_default(fake_diacnet_1):
    d = Diacritizer(model="diacnet-1.1", lang="yo")
    assert d.aligned is False
    assert d.restore("O so fun") == "Ó sọ fún?"           # raw, '?' and all, as before
    assert fake_diacnet_1.prompts == ["<yor> O so fun"]    # 1.x prompt format untouched


def test_diacnet_1_1_can_opt_in_to_alignment(fake_diacnet_1):
    d = Diacritizer(model="diacnet-1.1", lang="yo")
    assert d.restore("O so fun", aligned=True) == "Ó sọ fún"
    d2 = Diacritizer(model="diacnet-1.1", lang="yo", aligned=True)
    assert d2.restore("O so fun") == "Ó sọ fún"


def test_diacnet_1_1_still_rejects_arabic_and_the_2_0_options(fake_diacnet_1):
    d = Diacritizer(model="diacnet-1.1", lang="yo")
    with pytest.raises(ValueError, match="hints"):
        d.restore("O so", hints={"so": "say"})
    with pytest.raises(ValueError, match="case_endings"):
        d.restore("O so", case_endings=False)
    with pytest.raises(ValueError, match="Unsupported language"):
        Diacritizer(model="diacnet-1.1", lang="ar").restore("وهذا")
    with pytest.raises(ValueError, match="case_endings"):
        Diacritizer(model="diacnet-1.1", case_endings=False)


# =========================================================================== #
# Real checkpoints — opt in with: pytest -m slow   (needs an HF token)
# =========================================================================== #

def _golden_ids():
    return [f"{lang}{'-hint' if hint else ''}" for lang, hint, _, _ in GOLDEN]


@pytest.fixture(scope="module")
def real_diacnet():
    _needs_token()
    pytest.importorskip("torch")
    return Diacritizer(model="diacnet-2.0", device="auto")


@slow
@pytest.mark.parametrize("lang,hint,source,expected", GOLDEN, ids=_golden_ids())
def test_diacnet_2_golden_examples(real_diacnet, lang, hint, source, expected):
    """Exact strings from the reference GPU bf16 batched runs (aligned)."""
    out = real_diacnet.restore(source, lang=lang, hints=hint or None)
    assert out == expected


@slow
def test_diacnet_2_golden_examples_batched_together(real_diacnet):
    out = real_diacnet.restore_batch(
        [s for _, _, s, _ in GOLDEN],
        lang=[l for l, _, _, _ in GOLDEN],
        hints=[h or None for _, h, _, _ in GOLDEN])
    assert out == [e for _, _, _, e in GOLDEN]


@slow
def test_diacnet_2_aligned_output_strips_back_to_the_input(real_diacnet):
    for lang, _, source, _ in GOLDEN:
        assert strip_marks(real_diacnet.restore(source, lang=lang)) == strip_marks(source)


@slow
def test_diacnet_2_auto_tag_and_partial_input(real_diacnet):
    assert real_diacnet.restore("No pude sujetarme a la cuerda mas tiempo.") == \
        "No pude sujetarme a la cuerda más tiempo."
    assert real_diacnet.restore("Ki lo ṣe ti inu rẹ ko ni dun?", lang="yor") == \
        "Kí ló ṣe tí inú rẹ̀ kò ní dùn?"


@slow
def test_diacnet_2_arabic_without_case_endings():
    _needs_token()
    pytest.importorskip("torch")
    d = Diacritizer(model="diacnet-2.0", device="auto", case_endings=False)
    assert d.restore("وهذا قول مرغوب عنه .", lang="ara") == "وَهَذَا قَوْل مَرْغُوب عَنْه ."


@slow
def test_diacnet_mini_2_runs_and_keeps_the_invariant():
    _needs_token()
    pytest.importorskip("torch")
    d = Diacritizer(model="diacnet-mini-2.0", device="auto")
    for lang, hint, source, _ in GOLDEN:
        out = d.restore(source, lang=lang, hints=hint or None)
        assert out != source                                  # it added marks
        assert strip_marks(out) == strip_marks(source)


@slow
def test_diacnet_1_1_real_still_works_and_can_align():
    _needs_token()
    pytest.importorskip("torch")
    d = Diacritizer(model="diacnet-1.1", lang="yo")
    source = "O so fun ara re pe oun ko ni isoro kankan."
    raw = d.restore(source)
    assert raw
    assert strip_marks(d.restore(source, aligned=True)) == strip_marks(source)
