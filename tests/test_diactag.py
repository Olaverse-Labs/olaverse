"""
Tests for diactag-1.0 — the per-character diacritic tagger.

Three tiers:

  * pure — the vendored unicode/window/protected-span logic. No network, no
    torch. These are the invariants that make the architecture worth having, so
    they run everywhere.
  * wrapper — argument validation on ``Diacritizer`` and ``DiacTagDecoder``.
    Mocked; nothing is downloaded.
  * ``slow`` — the real checkpoint (~150MB). Excluded by default; run with
    ``pytest -m slow``. ``olaverse/diactag-1.0`` is a private repo, so these
    also need HF_TOKEN or a ``huggingface-cli login`` token.

The compliance test is the important one. It asserts the product claim: the
model cannot change, insert or delete a base character. It is run against
randomly generated text, and it must hold for every input, not on average.
"""

import random
import string

import pytest

from olaverse.nlp.diacritizer import (
    MODEL_REGISTRY,
    DiacTagDecoder,
    Diacritizer,
)
from olaverse.nlp._diactag.infer import PROTECTED_RE, plan_windows, protected_grapheme_flags
from olaverse.nlp._diactag.unicode_ops import (
    LANGS,
    SPEC_VERSION,
    base_char,
    compose_char,
    factorize_char,
    graphemes,
    normalize_lang,
    strip_diacritics,
)

slow = pytest.mark.slow

SAMPLES = {
    "yor": "ṣé ẹranko náà sì gbọ́ ọ?",
    "ibo": "Ndewo, kedu ka ị mere?",
    "hau": "ƙasar Hausa ɓarna ɗan yaƴi",
    "vie": "Cô ấy rất đảm đang.",
    "pol": "Zażółć gęślą jaźń, łódź i ćma.",
    "tur": "Iğdır'ın çığır açan şişli İstanbul.",
    "por": "Não é possível à mãe, coração.",
    "spa": "El niño comió mañana, ¿cuánto?",
    "fra": "Où êtes-vous ? Ça coûte cher.",
    "ita": "Andò a scuola, però non studiò.",
}


# =========================================================================== #
# Pure — unicode factorization
# =========================================================================== #

def test_spec_version_matches_shipped_label_space():
    """The checkpoint's labels.json carries a SPEC_VERSION that is checked on
    load. If the vendored spec drifts from what diactag-1.0 was built with,
    every load fails — so pin it here to catch the drift at test time instead.
    """
    assert SPEC_VERSION == "1.2.0"


@pytest.mark.parametrize("lang", LANGS)
def test_factorize_roundtrip(lang):
    for g in graphemes(SAMPLES[lang]):
        base, shape, tone = factorize_char(g, lang)
        assert compose_char(base, shape, tone) == g


@pytest.mark.parametrize("lang", LANGS)
def test_strip_preserves_grapheme_count(lang):
    """One grapheme in, one base character out. This alignment is what lets a
    per-character classifier be a valid model of the task at all."""
    assert len(strip_diacritics(SAMPLES[lang])) == len(graphemes(SAMPLES[lang]))


@pytest.mark.parametrize("lang", LANGS)
def test_strip_is_idempotent(lang):
    once = strip_diacritics(SAMPLES[lang])
    assert strip_diacritics(once) == once


def test_stacked_yoruba_marks_are_one_grapheme():
    """ọ̀ has no precomposed form — NFC leaves two codepoints. Iterating over
    str instead of graphemes misaligns every label after it."""
    word = "ọ̀rọ̀"
    assert len(word) > len(graphemes(word))
    assert len(graphemes(word)) == 3
    assert strip_diacritics(word) == "oro"


def test_base_char_is_ascii_for_special_letters():
    assert base_char("ł") == "l"      # Polish stroke
    assert base_char("đ") == "d"      # Vietnamese stroke
    assert base_char("ı") == "i"      # Turkish dotless i
    assert base_char("ƙ") == "k"      # Hausa hook


def test_dot_below_is_shape_in_yoruba_but_tone_in_vietnamese():
    """The reason shape and tone are separate heads: the same codepoint plays
    different grammatical roles per language."""
    _, yor_shape, yor_tone = factorize_char("ọ", "yor")
    _, vie_shape, vie_tone = factorize_char("ọ", "vie")
    assert yor_shape == ("DOT_BELOW",) and yor_tone == ""
    assert vie_shape == () and vie_tone == "̣"


def test_non_latin_script_passes_through_untouched():
    """Greek shares combining acute with Latin; without the script check its
    accents would be stripped."""
    assert strip_diacritics("Ελλάδα") == "Ελλάδα"


def test_normalize_lang_accepts_both_iso_forms():
    assert normalize_lang("yo") == "yor"
    assert normalize_lang("yor") == "yor"
    assert normalize_lang("pt-BR") == "por"
    assert normalize_lang("de") is None


# =========================================================================== #
# Pure — sliding windows and protected spans
# =========================================================================== #

@pytest.mark.parametrize("n", [1, 5, 383, 384, 385, 1000, 4096])
def test_plan_windows_partitions_exactly(n):
    """Every character is predicted exactly once, from the window in which it
    sits furthest from an edge. A gap silently drops characters; an overlap
    double-writes them."""
    covered = []
    for start, end, accept_start, accept_end in plan_windows(n, 384, 256):
        assert start <= accept_start < accept_end <= end
        covered.extend(range(accept_start, accept_end))
    assert covered == list(range(n))


def test_plan_windows_partitions_random_lengths():
    rng = random.Random(0)
    for _ in range(200):
        n = rng.randint(1, 5000)
        window = rng.choice([64, 128, 384])
        stride = rng.randint(max(1, window // 4), window)
        covered = []
        for _s, _e, a0, a1 in plan_windows(n, window, stride):
            covered.extend(range(a0, a1))
        assert covered == list(range(n)), (n, window, stride)


@pytest.mark.parametrize("text", [
    "visit https://ile-ife.com now",
    "email ade@olaverse.co.uk please",
    "ping @adebayo about it",
    "the MAX_RETRY_COUNT setting",
    "see olaverse.ng for more",
])
def test_protected_spans_are_flagged(text):
    flags = protected_grapheme_flags(graphemes(text))
    assert any(flags)
    assert not all(flags)              # surrounding prose is still restorable


def test_plain_prose_has_no_protected_spans():
    assert not any(protected_grapheme_flags(graphemes("se eranko naa si gbo o")))


def test_protected_re_is_available_without_the_training_module():
    """PROTECTED_RE is inlined into infer.py rather than vendored from
    diactag/data.py; this catches an upstream re-copy that drops it."""
    assert PROTECTED_RE.search("https://example.com") is not None


# =========================================================================== #
# Wrapper — registry and argument validation (no model loaded)
# =========================================================================== #

def test_registry_includes_diactag_and_diacnet_1_1():
    assert MODEL_REGISTRY["diactag-1.0"] == {"lang": "multi", "method": "diactag"}
    assert MODEL_REGISTRY["diacnet-1.1"] == {"lang": "multi", "method": "diacnet"}


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="not recognised"):
        Diacritizer(model="diactag-9.9")


def test_diactag_options_rejected_on_other_models():
    d = Diacritizer(model="diacnet-yor-viterbi")
    with pytest.raises(ValueError, match="min_confidence"):
        d.restore("se eranko", min_confidence=0.9)
    with pytest.raises(ValueError, match="return_details"):
        d.restore("se eranko", return_details=True)
    with pytest.raises(ValueError, match="detect_language"):
        d.detect_language("se eranko")
    with pytest.raises(ValueError, match="lang="):
        d.restore("se eranko", lang="yo")


def test_language_codes_are_validated_without_loading_weights():
    decoder = DiacTagDecoder.__new__(DiacTagDecoder)   # no download
    assert decoder.normalize_language("yo") == "yor"
    assert decoder.normalize_language("vie") == "vie"
    assert decoder.normalize_language(None) is None
    with pytest.raises(ValueError, match="Unsupported language 'de'"):
        decoder.normalize_language("de")


def test_onnx_backend_refuses_to_guess_the_language():
    """The shipped ONNX export omits the LID head, and an unguarded auto-detect
    there silently labels every input Yoruba."""
    decoder = DiacTagDecoder.__new__(DiacTagDecoder)
    decoder.supports_language_detection = False
    with pytest.raises(ValueError, match="ONNX export"):
        decoder._require_lid()


# --------------------------------------------------------------------------- #
# The ONNX adapter, against fake graphs — no onnxruntime, no artefacts.
#
# These pin the forward-compatibility contract: diactag's ExportWrapper is
# expected to grow the LID head, and this decoder must pick that up from the
# graph without a code change here.
# --------------------------------------------------------------------------- #

class _FakeIO:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    """Records what it was fed and returns one array per declared output."""

    def __init__(self, inputs, outputs):
        self._inputs = [_FakeIO(n) for n in inputs]
        self._outputs = [_FakeIO(n) for n in outputs]
        self.last_feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, _, feed):
        import numpy as np
        self.last_feed = feed
        shapes = {"shape_logits": (1, 3, 15), "tone_logits": (1, 3, 8),
                  "lid_logits": (1, 10)}
        return [np.zeros(shapes[o.name], dtype="float32") for o in self._outputs]


def _adapter(inputs, outputs):
    from olaverse.nlp.diacritizer import OnnxTaggerSession
    session = _FakeSession(inputs, outputs)
    return OnnxTaggerSession(session, n_langs=10), session


def test_onnx_adapter_detects_the_shipped_three_output_graph():
    adapter, _ = _adapter(["ids", "lang", "attn"],
                          ["shape_logits", "tone_logits"])
    assert adapter.has_lid is False


def test_onnx_adapter_picks_up_a_graph_that_exposes_the_lid_head():
    adapter, _ = _adapter(["ids", "lang", "lang_known", "attn"],
                          ["shape_logits", "tone_logits", "lid_logits"])
    assert adapter.has_lid is True


def test_onnx_adapter_rejects_lid_without_lang_known():
    """A lid_logits output on a graph with lang_known baked to 1 is not usable
    for detection — the head echoes whatever lang the caller passed."""
    adapter, _ = _adapter(["ids", "lang", "attn"],
                          ["shape_logits", "tone_logits", "lid_logits"])
    assert adapter.has_lid is False


@pytest.mark.parametrize("inputs", [
    ["ids", "lang", "attn"],
    ["ids", "lang", "lang_known", "attn"],
])
def test_onnx_adapter_feeds_only_declared_inputs(inputs):
    torch = pytest.importorskip("torch")
    outputs = ["shape_logits", "tone_logits"]
    if "lang_known" in inputs:
        outputs.append("lid_logits")
    adapter, session = _adapter(inputs, outputs)

    out = adapter(torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    assert set(session.last_feed) == set(inputs)
    assert set(out) == {"shape", "tone", "lid"}
    assert out["lid"].shape == (1, 10)


# =========================================================================== #
# Real checkpoint — opt in with: pytest -m slow
# =========================================================================== #

@slow
def test_diactag_real_restores_yoruba():
    d = Diacritizer(model="diactag-1.0", lang="yo")
    assert d.restore("se eranko naa si gbo o?") == "ṣé ẹranko náà sì gbọ́ ọ?"


@slow
def test_diactag_real_auto_detects_language():
    d = Diacritizer(model="diactag-1.0")
    lang, prob = d.detect_language("Lodz jest piekna")
    assert lang == "pol" and prob > 0.9
    assert d.restore("El nino esta en la casa") == "El niño está en la casa"


@slow
@pytest.mark.parametrize("lang", LANGS)
def test_diactag_real_compliance_holds_on_random_text(lang):
    """The product claim, asserted rather than assumed: whatever the model
    emits, stripping it returns the input's skeleton exactly."""
    rng = random.Random(1234)
    alphabet = string.ascii_letters + " .,?!'-0123456789"
    d = Diacritizer(model="diactag-1.0", lang=lang)
    for _ in range(5):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 300)))
        if not text.strip():
            continue
        assert strip_diacritics(d.restore(text)) == strip_diacritics(text)


@slow
def test_diactag_real_abstention_leaves_input_untouched():
    d = Diacritizer(model="diactag-1.0", lang="yor")
    src = "se eranko naa si gbo o?"
    committed = d.restore(src, min_confidence=0.0)
    cautious = d.restore(src, min_confidence=0.99)
    # Abstaining can only ever remove marks relative to committing to them.
    assert cautious != committed
    assert strip_diacritics(cautious) == strip_diacritics(src)


@slow
def test_diactag_real_per_character_confidence():
    d = Diacritizer(model="diactag-1.0", lang="yor")
    text, details = d.restore("se eranko naa", return_details=True)
    assert len(details) == len(graphemes(text))
    assert all(0.0 <= c.confidence <= 1.0 for c in details)


@slow
def test_diactag_real_leaves_urls_and_emails_alone():
    d = Diacritizer(model="diactag-1.0", lang="yor")
    out = d.restore("Visit https://ile-ife.com or email ade@ola.ng for eniyan")
    assert "https://ile-ife.com" in out
    assert "ade@ola.ng" in out
    assert "ènìyàn" in out             # surrounding prose still restored


@slow
def test_diactag_real_handles_documents_not_just_sentences():
    d = Diacritizer(model="diactag-1.0", lang="yor")
    doc = ("se eranko naa si gbo o. " * 40).strip()
    out = d.restore(doc)
    assert strip_diacritics(out) == strip_diacritics(doc)


@slow
def test_diactag_real_onnx_matches_pytorch():
    pytest.importorskip("onnxruntime")
    src = "se eranko naa si gbo o?"
    torch_out = Diacritizer(model="diactag-1.0", lang="yor").restore(src)
    onnx_out = Diacritizer(model="diactag-1.0", lang="yor", onnx=True).restore(src)
    assert onnx_out == torch_out


@slow
def test_diactag_real_onnx_carries_the_lid_head():
    """The published export emits lid_logits and takes lang_known, so the ONNX
    backend detects language rather than requiring lang=. An export without
    both is handled — it just refuses to guess — but is no longer what ships."""
    pytest.importorskip("onnxruntime")
    onnx = Diacritizer(model="diactag-1.0", onnx=True)
    torch_ = Diacritizer(model="diactag-1.0")
    assert onnx.neural_decoder.supports_language_detection

    for text in ["Lodz jest piekna", "Co ay rat dam dang",
                 "se eranko naa si gbo o?", "El nino esta en la casa"]:
        onnx_lang, onnx_p = onnx.detect_language(text)
        torch_lang, torch_p = torch_.detect_language(text)
        assert onnx_lang == torch_lang
        assert onnx_p == pytest.approx(torch_p, abs=1e-3)
