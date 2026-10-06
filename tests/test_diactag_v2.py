"""
Tests for diactag-2.0 and the label-space spec gate.

diactag-2.0 (spec 2.0.0) adds Arabic and ships with its own copy of the decoding
code, vendored as ``olaverse.nlp._diactag2`` next to the spec-1.x ``_diactag``
that diactag-1.0 keeps running on. ``labels.json`` records which generation a
checkpoint was built for, and the SDK routes on that field.

Tiers:

  * gate / pure — the spec selector and the vendored Arabic factorisation. No
    torch, no network.
  * tiny checkpoint — a throwaway randomly-initialised tagger, written to a temp
    directory and served through ``_diactag_fetch``. It checks the *plumbing*
    (routing, languages, case_endings, the strip invariant) end to end without
    downloading anything. It says nothing about accuracy.
  * ``slow`` — the real private checkpoints. Excluded by default
    (``pytest -m slow``), and skipped when Hugging Face is not reachable.
"""

import importlib
import json
import random

import pytest

from olaverse.nlp import diacritizer as dz
from olaverse.nlp._diactag_backend import (
    backend_for_labels,
    backend_for_spec,
    read_spec_version,
)
from olaverse.nlp.diacritizer import MODEL_REGISTRY, DiacTagDecoder, Diacritizer

slow = pytest.mark.slow

ARABIC_FULL = "ذَهَبَ الطَّالِبُ إِلَى الْمَدْرَسَةِ فِي الصَّبَاحِ"
YORUBA_FULL = "ṣé ẹranko náà sì gbọ́ ọ?"

# The Latin-script samples diactag-1.0's own tests use.
LATIN_SAMPLES = {
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


def _needs_hub():
    """Skip unless Hugging Face is reachable (the olaverse/* repos are public)."""
    import urllib.request
    try:
        urllib.request.urlopen(
            f"https://huggingface.co/olaverse/diactag-2.0/resolve/main/config.json", timeout=10).close()
    except Exception as exc:
        pytest.skip(f"Hugging Face is not reachable: {exc}")


# =========================================================================== #
# Helpers — build label spaces and tiny checkpoints
# =========================================================================== #

def _label_space(pkg: str):
    """A small corpus-derived label space built with the given vendored package."""
    labels = importlib.import_module(f"olaverse.nlp.{pkg}.labels")
    b = labels.LabelSpaceBuilder(min_char_count=1, min_combo_count=1)
    b.add(YORUBA_FULL, "yor")
    b.add("se eranko naa si gbo o", "yor")
    if pkg == "_diactag2":
        b.add(ARABIC_FULL, "ara")
        b.add("ذهب الطالب إلى المدرسة في الصباح", "ara")
        b.add("أَحْمَدُ يَقْرَأُ الْكِتَابَ", "ara")
    return b.build()


def _write_repo(root, name, pkg, ckpt, d_model=32):
    """Write labels.json + calibration.json + a tiny random checkpoint."""
    torch = pytest.importorskip("torch")
    repo = root / name
    repo.mkdir()
    ls = _label_space(pkg)
    ls.save(str(repo / "labels.json"))
    (repo / "calibration.json").write_text(json.dumps({"shared": 1.08}))
    model_mod = importlib.import_module(f"olaverse.nlp.{pkg}.model")
    cfg = model_mod.TaggerConfig(
        n_chars=ls.n_chars, n_shapes=ls.n_shapes, n_tones=ls.n_tones,
        n_langs=ls.n_langs, d_model=d_model, n_layers=1, n_heads=2, max_len=512)
    torch.manual_seed(0)
    model_mod.DiacTagger(cfg).save(str(repo / ckpt))
    return repo


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """Serve tiny diactag-1.0 and diactag-2.0 repos in place of the real ones."""
    _write_repo(tmp_path, "diactag-1.0", "_diactag", "ckpt_120000.pt")
    _write_repo(tmp_path, "diactag-2.0", "_diactag2", "ckpt_final.pt")

    def fetch(repo_id, filename, required=True):
        path = tmp_path / repo_id.rsplit("/", 1)[-1] / filename
        if path.exists():
            return str(path)
        if required:
            raise RuntimeError(f"no {filename} in fake {repo_id}")
        return None

    monkeypatch.setattr(dz, "_diactag_fetch", fetch)
    monkeypatch.setattr(dz, "_DIACTAG_CACHE", {})
    monkeypatch.setattr(dz, "_DIACTAG_LEXICON_CACHE", {})
    return tmp_path


# =========================================================================== #
# Spec gate — pure
# =========================================================================== #

def test_spec_versions_of_the_two_vendored_packages():
    assert backend_for_spec("1.2.0").spec_version == "1.2.0"
    assert backend_for_spec("2.0.0").spec_version == "2.0.0"


def test_spec_1x_routes_to_the_1x_code(tmp_path):
    path = tmp_path / "labels.json"
    _label_space("_diactag").save(str(path))
    assert read_spec_version(str(path)) == "1.2.0"
    backend = backend_for_labels(str(path))
    assert backend.package == "olaverse.nlp._diactag"
    ls = backend.module("labels").LabelSpace.load(str(path))
    assert ls.spec_version == "1.2.0"
    assert "ara" not in ls.langs


def test_spec_2_routes_to_the_new_code(tmp_path):
    path = tmp_path / "labels.json"
    _label_space("_diactag2").save(str(path))
    assert read_spec_version(str(path)) == "2.0.0"
    backend = backend_for_labels(str(path))
    assert backend.package == "olaverse.nlp._diactag2"
    ls = backend.module("labels").LabelSpace.load(str(path))
    assert ls.spec_version == "2.0.0"
    assert "ara" in ls.langs


def test_mismatched_spec_raises_instead_of_loading(tmp_path):
    """A 2.0.0 file must not load through the 1.x code, or vice versa — the
    strict check stays on, and the error says which specs are involved."""
    p1, p2 = tmp_path / "v1.json", tmp_path / "v2.json"
    _label_space("_diactag").save(str(p1))
    _label_space("_diactag2").save(str(p2))
    v1 = importlib.import_module("olaverse.nlp._diactag.labels").LabelSpace
    v2 = importlib.import_module("olaverse.nlp._diactag2.labels").LabelSpace
    with pytest.raises(ValueError, match=r"spec 2\.0\.0 but this code is 1\.2\.0"):
        v1.load(str(p2))
    with pytest.raises(ValueError, match=r"spec 1\.2\.0 but this code is 2\.0\.0"):
        v2.load(str(p1))


@pytest.mark.parametrize("spec,package", [
    ("1.1.0", "olaverse.nlp._diactag"),     # routed by major, then rejected exactly
    ("2.1.0", "olaverse.nlp._diactag2"),
])
def test_same_major_wrong_minor_is_rejected_by_the_exact_check(tmp_path, spec, package):
    path = tmp_path / "labels.json"
    pkg = package.rsplit(".", 1)[-1]
    _label_space(pkg).save(str(path))
    d = json.loads(path.read_text())
    d["spec_version"] = spec
    path.write_text(json.dumps(d))
    backend = backend_for_labels(str(path))
    assert backend.package == package
    with pytest.raises(ValueError, match="checkpoints are not portable"):
        backend.module("labels").LabelSpace.load(str(path))


@pytest.mark.parametrize("spec", ["3.0.0", "0.9.0", "banana"])
def test_unknown_major_spec_is_a_clear_error(spec):
    with pytest.raises(ValueError) as exc:
        backend_for_spec(spec)
    assert spec in str(exc.value)
    assert "1.x" in str(exc.value) and "2.x" in str(exc.value)
    assert "pip install -U olaverse" in str(exc.value)


def test_labels_without_a_spec_version_are_rejected(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"chars": []}))
    with pytest.raises(ValueError, match="spec_version"):
        read_spec_version(str(path))


def test_1x_code_is_unchanged_by_the_2_0_work():
    """diactag-1.0 must keep running on the 1.x code exactly as vendored."""
    from olaverse.nlp._diactag import unicode_ops
    assert unicode_ops.SPEC_VERSION == "1.2.0"
    assert "ara" not in unicode_ops.LANGS
    assert unicode_ops.normalize_lang("ar") is None


# =========================================================================== #
# Vendored 2.0 code — Arabic factorisation (pure)
# =========================================================================== #

@pytest.fixture(scope="module")
def uo2():
    return importlib.import_module("olaverse.nlp._diactag2.unicode_ops")


def test_2_0_label_constants(uo2):
    assert uo2.SPEC_VERSION == "2.0.0"
    assert uo2.LANGS[-1] == "ara" and len(uo2.LANGS) == 11
    assert uo2.normalize_lang("ar") == "ara"
    assert uo2.normalize_lang("ARA") == "ara"
    assert uo2.normalize_lang("yo") == "yor"


def test_arabic_marks_are_tone_or_shape(uo2):
    spec = uo2.SPEC["ara"]
    for m in (uo2.FATHA, uo2.DAMMA, uo2.KASRA, uo2.SUKUN,
              uo2.FATHATAN, uo2.DAMMATAN, uo2.KASRATAN):
        assert m in spec.tone_marks
    for m in (uo2.SHADDA, uo2.DAGGER_ALIF):
        assert m in spec.all_marks and m not in spec.tone_marks

    base, shape, tone = uo2.factorize_char("بَ", "ara")        # ba + fatha
    assert (base, shape, tone) == ("ب", (), uo2.FATHA)
    base, shape, tone = uo2.factorize_char("بّ", "ara")        # ba + shadda
    assert (base, shape, tone) == ("ب", ("SHADDA",), "")
    base, shape, tone = uo2.factorize_char("بَّ", "ara")       # shadda + fatha
    assert base == "ب" and shape == ("SHADDA",) and tone == uo2.FATHA


@pytest.mark.parametrize("letter", "أإآؤئ")
def test_hamza_letters_are_never_added_or_removed(uo2, letter):
    """Hamza letters are spelling: they stay whole letters (no NFD), so
    stripping neither removes nor invents a hamza."""
    assert uo2.base_char(letter) == letter
    assert uo2.base_char(letter + uo2.FATHA) == letter
    assert uo2.strip_diacritics(letter) == letter
    assert uo2.factorize_char(letter, "ara") == (letter, (), "")


def test_arabic_strips_back_to_the_bare_text(uo2):
    bare = "ذهب الطالب إلى المدرسة في الصباح"
    assert uo2.strip_diacritics(ARABIC_FULL) == bare
    assert uo2.strip_diacritics(bare) == bare


def test_arabic_roundtrips_through_factorize_and_compose(uo2):
    assert uo2.roundtrip_ok(ARABIC_FULL, "ara")


def test_arabic_marks_attach_to_their_letter(uo2):
    gs = uo2.graphemes(ARABIC_FULL)
    assert all(g[0] not in uo2.ARABIC_MARKS for g in gs)
    assert len(gs) == len(uo2.strip_diacritics(ARABIC_FULL))


def test_script_ok_checks_the_languages_own_script(uo2):
    assert uo2.script_ok("ذهب الطالب", "ara")
    assert not uo2.script_ok("hello world", "ara")
    assert uo2.script_ok("hello world", "yor")
    assert not uo2.script_ok("ذهب الطالب", "yor")


def test_latin_languages_still_factorise_the_same_way(uo2):
    """The Latin-script behaviour of the 2.0 code on the diactag-1.0 samples."""
    from olaverse.nlp._diactag import unicode_ops as uo1
    for lang, text in LATIN_SAMPLES.items():
        assert uo2.strip_diacritics(text) == uo1.strip_diacritics(text)
        for g in uo1.graphemes(text):
            assert uo2.factorize_char(g, lang) == uo1.factorize_char(g, lang)


# =========================================================================== #
# case_endings — the algorithm, against hand-built predictions
# =========================================================================== #

def _runtime_stub(ls, respect_existing=False):
    """An Arabic runtime with no model: just what _drop_case_endings touches."""
    infer = importlib.import_module("olaverse.nlp._diactag2.infer")
    rt = infer.Diacritizer.__new__(infer.Diacritizer)
    rt.ls = ls
    rt.cfg = infer.InferConfig(case_endings=False, respect_existing=respect_existing)
    return rt, infer


def _drop(text_with_marks, respect_existing=False, typed=None):
    """Apply _drop_case_endings to a 'perfect' prediction of text_with_marks."""
    uo2 = importlib.import_module("olaverse.nlp._diactag2.unicode_ops")
    ls = _label_space("_diactag2")
    rt, infer = _runtime_stub(ls, respect_existing)
    gold = uo2.graphemes(text_with_marks)
    gs = uo2.graphemes(typed if typed is not None else uo2.strip_diacritics(text_with_marks))
    shape_ids, _ = ls.labels_for(text_with_marks, "ara")
    out = list(gold)
    protected = [False] * len(gs)
    details = [infer.CharResult(c, 1.0, False, False) for c in out]
    rt._drop_case_endings(gs, out, shape_ids, protected, details)
    return "".join(out), details


def test_case_endings_false_drops_final_vowels_and_keeps_shadda():
    out, details = _drop(ARABIC_FULL)
    for word in out.split(" "):
        last = word[-1]
        assert last not in "ًٌٍَُِْ", word
    # a shadda on a word-final letter survives; only the vowel next to it goes
    shadda_out, _ = _drop("حَقٌّ")           # haqq: final qaf carries shadda + tanwin
    assert shadda_out == "حَقّ"
    # inner marks are untouched
    assert out.startswith("ذَهَب ") and "الطَّالِب" in out
    assert "".join(d.char for d in details) == out      # details follow the edits


def test_case_endings_false_leaves_a_mark_the_user_typed():
    """respect_existing: a word-final mark in the *input* is the user's intent."""
    typed = "ذهب الطالبُ"            # damma typed on the last letter of word 2
    out, _ = _drop("ذَهَبَ الطَّالِبُ", respect_existing=True, typed=typed)
    assert out.split(" ")[0].endswith("ب")            # untyped final vowel dropped
    assert out.split(" ")[1].endswith("بُ")           # typed one kept


def test_case_endings_ignores_non_arabic_words():
    out, _ = _drop("ذَهَبَ ok")        # the Latin word is not an Arabic letter run
    assert out.endswith(" ok")


# =========================================================================== #
# Tiny checkpoint — plumbing end to end
# =========================================================================== #

def test_registry_has_the_new_diactag():
    assert MODEL_REGISTRY["diactag-2.0"] == {"lang": "multi", "method": "diactag"}
    assert MODEL_REGISTRY["diactag-1.0"] == {"lang": "multi", "method": "diactag"}


def test_default_checkpoint_per_release():
    assert dz._diactag_default_ckpt("olaverse/diactag-1.0") == "ckpt_120000.pt"
    assert dz._diactag_default_ckpt("olaverse/diactag-2.0") == "ckpt_final.pt"
    # a fork keeps the historical default
    assert dz._diactag_default_ckpt("some-org/custom-tagger") == "ckpt_120000.pt"


def test_diactag_2_loads_through_the_new_code(fake_hub):
    d = DiacTagDecoder("olaverse/diactag-2.0")
    assert d._backend.package == "olaverse.nlp._diactag2"
    assert "ara" in d.LANGUAGES and len(d.LANGUAGES) == 11
    assert d._labels.spec_version == "2.0.0"


def test_diactag_1_still_loads_through_the_1x_code(fake_hub):
    d = DiacTagDecoder("olaverse/diactag-1.0")
    assert d._backend.package == "olaverse.nlp._diactag"
    assert "ara" not in d.LANGUAGES and len(d.LANGUAGES) == 10
    assert d._labels.spec_version == "1.2.0"
    # and it still restores text, with the invariant intact
    out = d.decode("se eranko naa si gbo o?", lang="yo")
    uo1 = importlib.import_module("olaverse.nlp._diactag.unicode_ops")
    assert uo1.strip_diacritics(out) == "se eranko naa si gbo o?"


@pytest.mark.parametrize("code", ["ara", "ar", "AR"])
def test_diactag_2_accepts_arabic_codes(fake_hub, code):
    assert DiacTagDecoder("olaverse/diactag-2.0").normalize_language(code) == "ara"


@pytest.mark.parametrize("code", ["ara", "ar"])
def test_diactag_1_rejects_arabic(fake_hub, code):
    with pytest.raises(ValueError, match="Unsupported language"):
        DiacTagDecoder("olaverse/diactag-1.0").normalize_language(code)


def test_unsupported_language_message_names_the_model(fake_hub):
    with pytest.raises(ValueError, match="for diactag-2.0"):
        DiacTagDecoder("olaverse/diactag-2.0").normalize_language("de")
    with pytest.raises(ValueError, match="for diactag-1.0"):
        DiacTagDecoder("olaverse/diactag-1.0").normalize_language("de")


def test_arabic_output_strips_back_to_the_input_exactly(fake_hub):
    uo2 = importlib.import_module("olaverse.nlp._diactag2.unicode_ops")
    d = DiacTagDecoder("olaverse/diactag-2.0")
    bare = "ذهب الطالب إلى المدرسة في الصباح"
    for text in (bare, "أحمد يقرأ الكتاب", "آمن المؤمنون وئد الأمل"):
        for lang in ("ara", "ar"):
            out = d.decode(text, lang=lang)
            assert uo2.strip_diacritics(out) == text
            assert uo2.strip_diacritics(out) == uo2.strip_diacritics(text)


def test_arabic_random_text_never_changes_a_letter(fake_hub):
    uo2 = importlib.import_module("olaverse.nlp._diactag2.unicode_ops")
    d = DiacTagDecoder("olaverse/diactag-2.0")
    rng = random.Random(7)
    letters = "ابتثجحخدذرزسشصضطظعغفقكلمنهويأإآؤئءةى"
    for _ in range(15):
        text = " ".join("".join(rng.choice(letters) for _ in range(rng.randint(1, 7)))
                        for _ in range(rng.randint(1, 8)))
        assert uo2.strip_diacritics(d.decode(text, lang="ara")) == text


def test_case_endings_false_leaves_no_word_final_harakat(fake_hub):
    d = DiacTagDecoder("olaverse/diactag-2.0", case_endings=False)
    text = "ذهب الطالب إلى المدرسة في الصباح وأحمد يقرأ الكتاب"
    vowels = set("ًٌٍَُِْ")
    for _ in range(2):                       # second call hits the cached runtime
        out = d.decode(text, lang="ara")
        for word in out.split(" "):
            assert word[-1] not in vowels, word
    # the per-call override works in both directions
    out = d.decode(text, lang="ara", case_endings=True)
    assert out != ""
    d_default = DiacTagDecoder("olaverse/diactag-2.0")
    out = d_default.decode(text, lang="ara", case_endings=False)
    for word in out.split(" "):
        assert word[-1] not in vowels, word


def test_case_endings_is_a_noop_for_other_languages(fake_hub):
    d_on = DiacTagDecoder("olaverse/diactag-2.0", case_endings=True)
    d_off = DiacTagDecoder("olaverse/diactag-2.0", case_endings=False)
    text = "se eranko naa si gbo o?"
    assert d_on.decode(text, lang="yor") == d_off.decode(text, lang="yor")


def test_case_endings_false_is_rejected_on_diactag_1(fake_hub):
    with pytest.raises(ValueError, match="case_endings only applies to Arabic"):
        DiacTagDecoder("olaverse/diactag-1.0", case_endings=False)
    d = DiacTagDecoder("olaverse/diactag-1.0")
    with pytest.raises(ValueError, match="case_endings only applies to Arabic"):
        d.decode("se eranko", lang="yor", case_endings=False)


def test_wrapper_exposes_case_endings_and_arabic(fake_hub):
    uo2 = importlib.import_module("olaverse.nlp._diactag2.unicode_ops")
    d = Diacritizer(model="diactag-2.0", lang="ar", case_endings=False)
    assert d.diactag_lang == "ara"
    text = "ذهب الطالب إلى المدرسة"
    out = d.restore(text)
    assert uo2.strip_diacritics(out) == text
    vowels = set("ًٌٍَُِْ")
    assert all(w[-1] not in vowels for w in out.split(" "))
    # per-call override, and details still come back
    out, details = d.restore(text, case_endings=True, return_details=True)
    assert len(details) == len(uo2.graphemes(text))


def test_wrapper_rejects_arabic_and_case_endings_on_diactag_1(fake_hub):
    with pytest.raises(ValueError, match="Unsupported language"):
        Diacritizer(model="diactag-1.0", lang="ar")
    with pytest.raises(ValueError, match="case_endings only applies to Arabic"):
        Diacritizer(model="diactag-1.0", case_endings=False)


def test_case_endings_rejected_on_models_that_have_no_arabic():
    with pytest.raises(ValueError, match="case_endings"):
        Diacritizer(model="diacnet-yor-viterbi", case_endings=False)
    with pytest.raises(ValueError, match="case_endings"):
        Diacritizer(model="diacnet-yor-viterbi").restore("se eranko", case_endings=False)


def test_default_1_0_ckpt_resolution_is_unchanged_for_existing_callers(fake_hub):
    """DiacTagDecoder() with no ckpt= keeps loading diactag-1.0's checkpoint."""
    d = DiacTagDecoder()
    assert d.model_name == "olaverse/diactag-1.0"
    assert d._backend.package == "olaverse.nlp._diactag"


def test_onnx_adapter_matches_an_11_language_graph():
    """The ONNX adapter is spec-agnostic: it is told n_langs by the label space."""
    class _IO:
        def __init__(self, name):
            self.name = name

    class _Session:
        def get_inputs(self):
            return [_IO(n) for n in ("ids", "lang", "lang_known", "attn")]

        def get_outputs(self):
            return [_IO(n) for n in ("shape_logits", "tone_logits", "lid_logits")]

    adapter = dz.OnnxTaggerSession(_Session(), n_langs=11)
    assert adapter.n_langs == 11 and adapter.has_lid


# =========================================================================== #
# Real checkpoint — opt in with: pytest -m slow   (downloads the models)
# =========================================================================== #

@pytest.fixture(scope="module")
def real_tagger():
    _needs_hub()
    pytest.importorskip("torch")
    return DiacTagDecoder("olaverse/diactag-2.0")


@slow
def test_diactag_2_real_label_space():
    _needs_hub()
    path = dz._diactag_fetch("olaverse/diactag-2.0", "labels.json")
    assert read_spec_version(path) == "2.0.0"
    ls = backend_for_labels(path).module("labels").LabelSpace.load(path)
    assert (ls.n_chars, ls.n_shapes, ls.n_tones) == (986, 18, 15)
    assert "ara" in ls.langs and len(ls.langs) == 11


@slow
def test_diactag_2_real_restores_yoruba(real_tagger):
    assert real_tagger.decode("se eranko naa si gbo o?", lang="yor") == YORUBA_FULL


@slow
def test_diactag_2_real_arabic_strips_back_to_input(real_tagger):
    uo2 = importlib.import_module("olaverse.nlp._diactag2.unicode_ops")
    text = "ذهب الطالب إلى المدرسة في الصباح"
    out = real_tagger.decode(text, lang="ara")
    assert out != text                       # it actually added marks
    assert uo2.strip_diacritics(out) == text


@slow
def test_diactag_2_real_case_endings_false_has_no_final_harakat(real_tagger):
    text = "ذهب الطالب إلى المدرسة في الصباح"
    vowels = set("ًٌٍَُِْ")
    full = real_tagger.decode(text, lang="ara", case_endings=True)
    nocase = real_tagger.decode(text, lang="ara", case_endings=False)
    assert any(w[-1] in vowels for w in full.split(" "))
    assert all(w[-1] not in vowels for w in nocase.split(" "))


@slow
def test_diactag_2_real_auto_detects_arabic(real_tagger):
    lang, p = real_tagger.detect_language("ذهب الطالب إلى المدرسة في الصباح")
    assert lang == "ara"


@slow
def test_diactag_2_real_onnx_matches_pytorch():
    _needs_hub()
    pytest.importorskip("onnxruntime")
    pt = DiacTagDecoder("olaverse/diactag-2.0")
    ox = DiacTagDecoder("olaverse/diactag-2.0", onnx=True)
    for text, lang in (("se eranko naa si gbo o?", "yor"),
                       ("ذهب الطالب إلى المدرسة", "ara")):
        a, b = pt.decode(text, lang=lang), ox.decode(text, lang=lang)
        agree = sum(x == y for x, y in zip(a, b)) / max(1, len(a))
        assert agree > 0.9, (a, b)
