"""
Tests for lid-lite-608 (fastText) and lid-neural-608 (mmBERT) — 608-language
identification with a coverage mode and a traffic mode.

Tiers:

  * mocked — both classes against a fake fastText model / fake tokenizer and
    model: mode logic, label handling, whitespace, empty input, device and dtype,
    error messages. Nothing is downloaded. The neural tests use the real torch.
  * ``slow`` — the real public checkpoints (37 MB and ~560 MB). Excluded by
    default (``pytest -m slow``) so no automatic check downloads them, and skipped
    when Hugging Face is not reachable.
"""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from olaverse.nlp import language_detection as ld
from olaverse.nlp.language_detection import (
    LIDLite608,
    LIDNeural25,
    LIDNeural5,
    LIDNeural608,
)

slow = pytest.mark.slow

# alpha * log_prior_ratio: the traffic bias bbb gets over aaa
PRIORS = {"alpha": 0.2, "log_prior_ratio": {"aaa_Latn": 0.0, "bbb_Latn": 5.0, "ccc_Latn": -5.0,
                                           "zxx_Zxxx": 0.0}}


def _needs_hub():
    import urllib.request
    try:
        urllib.request.urlopen(
            "https://huggingface.co/olaverse/lid-lite-608/resolve/main/priors.json",
            timeout=10).close()
    except Exception as exc:
        pytest.skip(f"Hugging Face is not reachable: {exc}")


@pytest.fixture
def priors_file(tmp_path):
    path = tmp_path / "priors.json"
    path.write_text(json.dumps(PRIORS))
    return str(path)


# =========================================================================== #
# Shared
# =========================================================================== #

def test_exported_from_the_package_roots():
    import olaverse
    import olaverse.nlp
    assert olaverse.LIDLite608 is LIDLite608 is olaverse.nlp.LIDLite608
    assert olaverse.LIDNeural608 is LIDNeural608 is olaverse.nlp.LIDNeural608


@pytest.mark.parametrize("cls", [LIDLite608, LIDNeural608])
def test_invalid_mode_is_rejected(cls):
    with pytest.raises(ValueError, match="mode must be one of"):
        cls(mode="bogus")


@pytest.mark.parametrize("cls", [LIDLite608, LIDNeural608])
def test_default_mode_is_coverage(cls):
    assert cls().mode == "coverage"
    assert cls(mode="traffic").mode == "traffic"


def test_repo_ids():
    assert LIDLite608.REPO_ID == "olaverse/lid-lite-608"
    assert LIDNeural608().model_name == "olaverse/lid-neural-608"


# =========================================================================== #
# LIDLite608 — against a fake fastText model
# =========================================================================== #

class _FakeFastText:
    """Returns canned candidates and records what it was asked."""

    def __init__(self, candidates):
        self.candidates = candidates          # [(label, prob)], best first
        self.calls = []

    def predict(self, text, k=1):
        self.calls.append((text, k))
        picked = self.candidates if k < 0 else self.candidates[:k]
        return tuple(l for l, _ in picked), tuple(p for _, p in picked)


def _lite(candidates, mode, priors_file, model=None):
    model = model or _FakeFastText(candidates)
    fake_fasttext = MagicMock()
    fake_fasttext.load_model.return_value = model

    def fake_path(filename, repo_id=None):
        assert repo_id == "olaverse/lid-lite-608"
        return {"model.ftz": "/fake/model.ftz", "priors.json": priors_file}[filename]

    with patch.dict("sys.modules", {"fasttext": fake_fasttext}), \
         patch.object(ld, "get_model_path", side_effect=fake_path):
        lid = LIDLite608(mode=mode)
        lid.load()
    return lid, model


CANDIDATES = [("__label__aaa_Latn", 0.60), ("__label__bbb_Latn", 0.39), ("__label__ccc_Latn", 0.01)]


def test_lite_import_error_names_the_extra():
    with patch.dict("sys.modules", {"fasttext": None}):
        with pytest.raises(ImportError, match=r"olaverse\[lid\]"):
            LIDLite608().load()


def test_lite_loads_the_model_file_from_its_repo(priors_file):
    lid, _ = _lite(CANDIDATES, "coverage", priors_file)
    assert lid._model is not None


def test_lite_coverage_returns_the_models_top_label_without_the_prefix(priors_file):
    lid, model = _lite(CANDIDATES, "coverage", priors_file)
    assert lid.predict("some text") == "aaa_Latn"
    assert model.calls[-1][1] == 1                        # asks for one candidate only


def test_lite_coverage_probabilities_are_the_models_own(priors_file):
    lid, _ = _lite(CANDIDATES, "coverage", priors_file)
    probs = lid.predict_proba("some text")
    assert probs == {"aaa_Latn": 0.60, "bbb_Latn": 0.39, "ccc_Latn": 0.01}
    assert list(lid.predict_proba("some text", top_k=2)) == ["aaa_Latn", "bbb_Latn"]


def test_lite_traffic_reranks_with_the_priors(priors_file):
    """bbb is 0.39 vs 0.60, but its traffic bias (0.2 * 5 = 1.0) outweighs
    log(0.60 / 0.39) = 0.43, so traffic mode answers bbb where coverage answers aaa."""
    coverage, _ = _lite(CANDIDATES, "coverage", priors_file)
    traffic, model = _lite(CANDIDATES, "traffic", priors_file)
    assert coverage.predict("x") == "aaa_Latn"
    assert traffic.predict("x") == "bbb_Latn"
    assert model.calls[-1][1] == LIDLite608.TRAFFIC_CANDIDATES == 40


def test_lite_traffic_probabilities_are_normalised_and_ordered(priors_file):
    traffic, _ = _lite(CANDIDATES, "traffic", priors_file)
    probs = traffic.predict_proba("x")
    assert sum(probs.values()) == pytest.approx(1.0)
    assert list(probs) == ["bbb_Latn", "aaa_Latn", "ccc_Latn"]
    assert max(probs, key=probs.get) == traffic.predict("x")
    assert list(traffic.predict_proba("x", top_k=1)) == ["bbb_Latn"]


def test_lite_traffic_bias_matches_the_card_formula(priors_file):
    """score = log(max(p, 1e-12)) + alpha * log_prior_ratio, argmax over candidates."""
    import math
    cands = [("__label__aaa_Latn", 0.5), ("__label__ccc_Latn", 0.5), ("__label__zzz_Latn", 0.0)]
    traffic, _ = _lite(cands, "traffic", priors_file)
    scores = {l[9:]: math.log(max(p, 1e-12)) + 0.2 * PRIORS["log_prior_ratio"].get(l[9:], 0.0)
              for l, p in cands}
    assert traffic.predict("x") == max(scores, key=scores.get) == "aaa_Latn"
    # a label missing from the priors gets no bias rather than raising
    assert "zzz_Latn" in traffic.predict_proba("x")


def test_lite_collapses_newlines_and_runs_of_whitespace(priors_file):
    """fastText reads one line; a newline would silently drop the rest."""
    lid, model = _lite(CANDIDATES, "coverage", priors_file)
    lid.predict("Habari\nza   asubuhi\r\n ")
    assert model.calls[-1][0] == "Habari za asubuhi"


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_lite_empty_input_is_a_clear_error(priors_file, text):
    lid, _ = _lite(CANDIDATES, "coverage", priors_file)
    with pytest.raises(ValueError, match="text is empty"):
        lid.predict(text)
    with pytest.raises(ValueError, match="text is empty"):
        lid.predict_proba(text)


def test_lite_no_candidates_means_noise(priors_file):
    for mode in ("coverage", "traffic"):
        lid, _ = _lite([], mode, priors_file)
        assert lid.predict("%%%") == "zxx_Zxxx"
        assert lid.predict_proba("%%%") == {}


def test_lite_survives_the_numpy_2_predict_failure(priors_file):
    """fasttext-wheel's predict() does np.array(copy=False), which NumPy >= 2
    rejects. The class falls back to the binding underneath."""
    class Numpy2Model:
        def __init__(self):
            self.f = MagicMock()
            self.f.predict.return_value = [(0.9, "__label__yor_Latn"), (0.1, "__label__ibo_Latn")]

        def predict(self, text, k=1):
            raise ValueError("Unable to avoid copy while creating an array as requested.")

    model = Numpy2Model()
    lid, _ = _lite(None, "coverage", priors_file, model=model)
    assert lid.predict("Ẹ kú àárọ̀") == "yor_Latn"
    assert lid.predict_proba("Ẹ kú àárọ̀") == {"yor_Latn": 0.9, "ibo_Latn": 0.1}
    text, k, threshold, on_error = model.f.predict.call_args[0]
    assert text == "Ẹ kú àárọ̀\n" and k == -1 and threshold == 0.0 and on_error == "strict"


def test_lite_predict_batch(priors_file):
    lid, _ = _lite(CANDIDATES, "coverage", priors_file)
    assert lid.predict_batch(["a", "b"]) == ["aaa_Latn", "aaa_Latn"]
    assert lid.predict_batch([]) == []


def test_lite_coverage_mode_does_not_fetch_the_priors():
    seen = []

    def fake_path(filename, repo_id=None):
        seen.append(filename)
        return "/fake/" + filename

    fake_fasttext = MagicMock()
    fake_fasttext.load_model.return_value = _FakeFastText(CANDIDATES)
    with patch.dict("sys.modules", {"fasttext": fake_fasttext}), \
         patch.object(ld, "get_model_path", side_effect=fake_path):
        LIDLite608().load()
    assert seen == ["model.ftz"]


# =========================================================================== #
# LIDNeural608 — real torch, fake transformers
# =========================================================================== #

LABELS = ["aaa_Latn", "bbb_Latn", "ccc_Latn", "zxx_Zxxx"]
# logits: aaa 0.60, bbb 0.39 (after softmax-ish); traffic bias (+1.0) lifts bbb
LOGITS = [3.0, 2.57, -1.0, -5.0]


class _Enc(dict):
    def to(self, device):
        self.device = device
        return self


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, return_tensors=None, truncation=None, max_length=None, padding=None):
        self.calls.append({"texts": texts, "truncation": truncation,
                           "max_length": max_length, "padding": padding})
        n = len(texts) if isinstance(texts, list) else 1
        import torch
        return _Enc(input_ids=torch.zeros(n, 3, dtype=torch.long))


class _FakeModel:
    def __init__(self, logits_rows=None):
        import torch
        self.config = types.SimpleNamespace(id2label={str(i): l for i, l in enumerate(LABELS)})
        self.rows = logits_rows
        self.moved_to = None
        self.evaluated = False
        self._torch = torch

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, input_ids=None, **kw):
        n = input_ids.shape[0]
        rows = self.rows or [LOGITS] * n
        return types.SimpleNamespace(logits=self._torch.tensor(rows[:n], dtype=self._torch.float32))


@pytest.fixture
def fake_hf(monkeypatch, priors_file):
    pytest.importorskip("torch")
    tok, model = _FakeTokenizer(), _FakeModel()
    seen = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kw):
            seen["tokenizer"] = name
            return tok

    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(name, **kw):
            seen["model"] = (name, kw)
            return model

    mod = types.ModuleType("transformers")
    mod.__version__ = "5.19.0"
    mod.AutoTokenizer = AutoTokenizer
    mod.AutoModelForSequenceClassification = AutoModelForSequenceClassification
    monkeypatch.setitem(sys.modules, "transformers", mod)
    monkeypatch.setattr(
        ld, "get_model_path",
        lambda filename, repo_id=None: priors_file if filename == "priors.json" else "/fake/" + filename)
    return types.SimpleNamespace(tok=tok, model=model, seen=seen, module=mod)


def test_neural_import_error_names_the_extra():
    with patch.dict("sys.modules", {"transformers": None}):
        with pytest.raises(ImportError, match=r"olaverse\[deeplearning\]"):
            LIDNeural608().load()


def test_neural_reads_512_tokens_while_the_older_models_keep_128():
    assert LIDNeural608.max_length == 512
    assert LIDNeural25.max_length == 128 and LIDNeural5.max_length == 128


def test_neural_loads_with_the_sequence_classifier_and_takes_classes_from_the_config(fake_hf):
    lid = LIDNeural608(device="cpu")
    lid.load()
    assert fake_hf.seen["tokenizer"] == fake_hf.seen["model"][0] == "olaverse/lid-neural-608"
    assert lid.classes == LABELS
    assert fake_hf.model.evaluated and lid._loaded


def test_neural_coverage_is_plain_softmax(fake_hf):
    lid = LIDNeural608(device="cpu")
    probs = lid.predict_proba("some text")
    assert list(probs) == LABELS
    assert sum(probs.values()) == pytest.approx(1.0)
    assert lid.predict("some text") == "aaa_Latn"


def test_neural_traffic_reranks_with_the_priors(fake_hf):
    """bbb's traffic bias (0.2 * 5 = 1.0) outweighs the 0.43 logit gap."""
    coverage = LIDNeural608(device="cpu")
    traffic = LIDNeural608(device="cpu", mode="traffic")
    assert coverage.predict("x") == "aaa_Latn"
    assert traffic.predict("x") == "bbb_Latn"
    probs = traffic.predict_proba("x")
    assert sum(probs.values()) == pytest.approx(1.0)
    assert max(probs, key=probs.get) == "bbb_Latn"
    # bias follows the model's own label order, including the noise class
    assert traffic._bias.tolist() == pytest.approx([0.0, 1.0, -1.0, 0.0])


def test_neural_traffic_matches_the_card_formula(fake_hf):
    import torch
    lid = LIDNeural608(device="cpu", mode="traffic")
    got = lid.predict_proba("x")
    logp = torch.log_softmax(torch.tensor(LOGITS), -1) + torch.tensor([0.0, 1.0, -1.0, 0.0])
    want = torch.softmax(logp, -1).tolist()
    assert [got[l] for l in LABELS] == pytest.approx(want)


def test_neural_tokenizer_gets_collapsed_whitespace_and_the_512_limit(fake_hf):
    lid = LIDNeural608(device="cpu")
    lid.predict("Habari\nza   asubuhi \r\n")
    call = fake_hf.tok.calls[-1]
    assert call["texts"] == "Habari za asubuhi"
    assert call["truncation"] is True and call["max_length"] == 512


def test_neural_batch_pads_and_collapses_each_text(fake_hf):
    fake_hf.model.rows = [[3.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 3.0, 0.0]]
    lid = LIDNeural608(device="cpu")
    assert lid.predict_batch(["a  b", "c\nd", "e"]) == ["aaa_Latn", "bbb_Latn", "ccc_Latn"]
    call = fake_hf.tok.calls[-1]
    assert call["texts"] == ["a b", "c d", "e"] and call["padding"] is True
    assert call["max_length"] == 512
    assert lid.predict_batch([]) == []


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_neural_empty_input_is_a_clear_error(fake_hf, text):
    lid = LIDNeural608(device="cpu")
    with pytest.raises(ValueError, match="text is empty"):
        lid.predict(text)
    with pytest.raises(ValueError, match="text is empty"):
        lid.predict_batch(["fine", text])


def test_neural_uses_bf16_on_cuda_and_float32_elsewhere(fake_hf):
    import torch
    LIDNeural608(device="cuda").load()
    assert fake_hf.seen["model"][1]["dtype"] is torch.bfloat16
    assert fake_hf.model.moved_to == "cuda"
    LIDNeural608(device="cuda:1").load()
    assert fake_hf.seen["model"][1]["dtype"] is torch.bfloat16
    LIDNeural608(device="cpu").load()
    assert "dtype" not in fake_hf.seen["model"][1]
    LIDNeural608(device="mps").load()
    assert "dtype" not in fake_hf.seen["model"][1]


def test_neural_device_auto_picks_cuda_only_when_available(fake_hf, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    lid = LIDNeural608()
    lid.load()
    assert lid._device == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    lid = LIDNeural608(device="auto")
    lid.load()
    assert lid._device == "cuda"


def test_neural_a_load_failure_on_an_old_transformers_says_what_to_upgrade(fake_hf):
    def boom(name, **kw):
        raise ValueError("model type `modernbert` not recognised")
    fake_hf.module.AutoModelForSequenceClassification.from_pretrained = staticmethod(boom)
    fake_hf.module.__version__ = "4.46.0"
    with pytest.raises(RuntimeError, match=r"transformers>=5\.14") as exc:
        LIDNeural608(device="cpu").load()
    assert "4.46.0" in str(exc.value) and "modernbert" in str(exc.value)


def test_neural_a_load_failure_on_a_current_transformers_is_not_rewritten(fake_hf):
    def boom(name, **kw):
        raise OSError("network down")
    fake_hf.module.AutoModelForSequenceClassification.from_pretrained = staticmethod(boom)
    with pytest.raises(OSError, match="network down"):
        LIDNeural608(device="cpu").load()


def test_neural_traffic_fetches_the_priors_only_in_traffic_mode(fake_hf, monkeypatch):
    seen = []
    monkeypatch.setattr(ld, "get_model_path",
                        lambda filename, repo_id=None: seen.append((filename, repo_id)) or "/fake/x")
    LIDNeural608(device="cpu").load()
    assert seen == []                                # coverage: nothing beyond transformers' own files


# =========================================================================== #
# Existing neural classes — untouched by the shared-base changes
# =========================================================================== #

def test_existing_neural_classes_still_have_no_device_and_the_same_defaults():
    assert LIDNeural25()._device is None
    assert LIDNeural5()._device is None


# =========================================================================== #
# Real checkpoints — opt in with: pytest -m slow   (public repos, downloads)
# =========================================================================== #

CARD = [("Ẹ kú àárọ̀, ṣé dáadáa ni?", "yor_Latn"), ("Habari za asubuhi", "swh_Latn")]


@pytest.fixture(scope="module")
def real_lite():
    _needs_hub()
    pytest.importorskip("fasttext")
    lid = LIDLite608()
    lid.load()
    return lid


@pytest.fixture(scope="module")
def real_neural():
    _needs_hub()
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    lid = LIDNeural608(device="cpu")
    lid.load()
    return lid


@slow
@pytest.mark.parametrize("text,label", CARD)
def test_lite_real_matches_the_model_card(real_lite, text, label):
    assert real_lite.predict(text) == label


@slow
def test_lite_real_traffic_mode_matches_the_model_card():
    _needs_hub()
    pytest.importorskip("fasttext")
    assert LIDLite608(mode="traffic").predict("Bonjour mon ami") == "fra_Latn"


@slow
def test_lite_real_probabilities_and_noise_class(real_lite):
    probs = real_lite.predict_proba("Habari za asubuhi")
    assert len(probs) == 609 and max(probs, key=probs.get) == "swh_Latn"
    assert real_lite.predict("12345 67890") == "zxx_Zxxx"
    assert real_lite.predict("Habari\nza\nasubuhi") == "swh_Latn"


@slow
@pytest.mark.parametrize("text,label", CARD + [("Bonjour mon ami", "fra_Latn")])
def test_neural_real_matches_the_model_card(real_neural, text, label):
    assert real_neural.predict(text) == label


@slow
def test_neural_real_traffic_mode_matches_the_model_card():
    _needs_hub()
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    lid = LIDNeural608(device="cpu", mode="traffic")
    assert lid.predict_batch(["Good morning", "Thank you very much"]) == ["eng_Latn", "eng_Latn"]


@slow
def test_neural_real_batch_probabilities_and_noise_class(real_neural):
    assert len(real_neural.classes) == 609
    assert real_neural.predict_batch(["Habari za asubuhi", "Mo fẹ́ lọ sí ọjà lónìí"]) == \
        ["swh_Latn", "yor_Latn"]
    probs = real_neural.predict_proba("Habari za asubuhi")
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert real_neural.predict("12345 67890") == "zxx_Zxxx"
