"""
test_new_models.py

Tests for the models added to the olaverse SDK on top of the existing
LIDLite5/LIDNeural5/Diacritizer/Tokenizer surface: LIDLite25, LIDNeural25,
LIDNeural5_1, diacnet-1.0 (DiacNetDecoder), the otk-bpe multilingual tokenizer
family, Reranker, Embedder, PrismUpscaler, and PrismSteganography.

All tests here are lightweight/mocked — no network access, no GPU, no heavy
optional dependencies actually installed.
"""

import pytest
from unittest.mock import patch, MagicMock

from olaverse.nlp.language_detection import LIDLite25, LIDNeural25, LIDNeural5_1
from olaverse.nlp.diacritizer import Diacritizer, MODEL_REGISTRY, DiacNetDecoder
from olaverse.nlp.tokenizer import Tokenizer
from olaverse.nlp.retrieval import Reranker, Embedder
from olaverse.vision import PrismUpscaler, PrismDenoiser, PrismSteganography
from olaverse.utils.downloader import get_model_path


# =========================================================================== #
# LIDLite25
# =========================================================================== #

def test_lid_lite_25_invalid_variant():
    with pytest.raises(ValueError):
        LIDLite25(variant="bogus")


def test_lid_lite_25_import_error():
    with patch.dict("sys.modules", {"fasttext": None}):
        detector = LIDLite25()
        with pytest.raises(ImportError) as exc_info:
            detector.load()
        assert "fasttext" in str(exc_info.value)


def test_lid_lite_25_mocked():
    mock_ft_model = MagicMock()
    mock_ft_model.predict.return_value = (
        ("__label__eng", "__label__fra"),
        (0.9, 0.1),
    )
    mock_fasttext = MagicMock()
    mock_fasttext.load_model.return_value = mock_ft_model

    with patch.dict("sys.modules", {"fasttext": mock_fasttext}), \
         patch("olaverse.nlp.language_detection.get_model_path", return_value="/fake/questions.bin"):
        detector = LIDLite25(variant="questions")
        detector.load()

        assert detector.predict("What causes ocean tides?") == "eng"
        probs = detector.predict_proba("What causes ocean tides?")
        assert probs["eng"] == pytest.approx(0.9)
        assert probs["fra"] == pytest.approx(0.1)


# =========================================================================== #
# LIDNeural25 / LIDNeural5_1
# =========================================================================== #

def test_lid_neural_25_invalid_variant():
    with pytest.raises(ValueError):
        LIDNeural25(variant="bogus")


def test_lid_neural_25_model_ids():
    assert LIDNeural25(variant="passages").model_name == "olaverse/lid-neural-25.1"
    assert LIDNeural25(variant="questions").model_name == "olaverse/lid-neural-25.2"


def test_lid_neural_51_default_model_name():
    assert LIDNeural5_1().model_name == "olaverse/lid-neural-5.1"


def test_lid_neural_25_import_error():
    with patch.dict("sys.modules", {"transformers": None}):
        detector = LIDNeural25()
        with pytest.raises(ImportError) as exc_info:
            detector.load()
        assert "transformers" in str(exc_info.value)


def _mocked_transformers_classifier(id2label, logits_probs):
    mock_transformers = MagicMock()
    mock_tokenizer_cls = MagicMock()
    mock_model_cls = MagicMock()

    mock_trained_tokenizer = MagicMock()
    mock_tokenizer_cls.from_pretrained.return_value = mock_trained_tokenizer
    mock_trained_tokenizer.return_value = {"input_ids": [1, 2, 3]}

    mock_trained_model = MagicMock()
    mock_model_cls.from_pretrained.return_value = mock_trained_model
    mock_trained_model.config.id2label = id2label

    mock_output = MagicMock()
    mock_output.logits = MagicMock()
    mock_trained_model.return_value = mock_output

    mock_transformers.AutoTokenizer = mock_tokenizer_cls
    mock_transformers.AutoModelForSequenceClassification = mock_model_cls

    mock_torch = MagicMock()
    mock_torch.no_grad = lambda: MagicMock()
    mock_softmax_val = MagicMock()
    mock_softmax_val.squeeze.return_value.tolist.return_value = logits_probs
    mock_torch.softmax.return_value = mock_softmax_val

    return mock_transformers, mock_torch


def test_lid_neural_51_mocked():
    id2label = {"0": "Hausa", "1": "Yoruba", "2": "Igbo", "3": "Nigerian Pidgin"}
    mock_transformers, mock_torch = _mocked_transformers_classifier(id2label, [0.95, 0.02, 0.02, 0.01])

    with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": mock_torch}):
        detector = LIDNeural5_1()
        detector.load()

        assert detector._loaded is True
        assert detector.classes == ["Hausa", "Yoruba", "Igbo", "Nigerian Pidgin"]
        assert detector.predict("Ina kwana?") == "Hausa"


def test_lid_neural_25_mocked_batch():
    id2label = {str(i): lbl for i, lbl in enumerate(["eng", "fra"])}
    mock_transformers, mock_torch = _mocked_transformers_classifier(id2label, [0.3, 0.7])
    # predict_proba_batch path uses tolist() directly on the softmax result (not squeeze())
    mock_torch.softmax.return_value.tolist.return_value = [[0.3, 0.7], [0.8, 0.2]]

    with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": mock_torch}):
        detector = LIDNeural25(variant="questions")
        detector.load()

        preds = detector.predict_batch(["bonjour", "hello"])
        assert preds == ["fra", "eng"]


# =========================================================================== #
# Diacritizer — diacnet-1.0 (DiacNetDecoder)
# =========================================================================== #

def test_diacritizer_registry_includes_diacnet_1_0():
    assert MODEL_REGISTRY["diacnet-1.0"] == {"lang": "multi", "method": "diacnet"}


def test_diacnet_decoder_unsupported_language():
    mock_tokenizer_cls = MagicMock()
    mock_model_cls = MagicMock()
    mock_transformers = MagicMock(AutoTokenizer=mock_tokenizer_cls, T5ForConditionalGeneration=mock_model_cls)

    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        decoder = DiacNetDecoder()
        with pytest.raises(ValueError, match="Unsupported language"):
            decoder.decode("some text", lang="zz")


def test_diacnet_decoder_mocked_decode():
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
    mock_tokenizer.decode.return_value = "ṣé ẹranko náà sì gbọ́ ọ?"

    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3, 4]]

    mock_tokenizer_cls = MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer))
    mock_model_cls = MagicMock(from_pretrained=MagicMock(return_value=mock_model))
    mock_transformers = MagicMock(AutoTokenizer=mock_tokenizer_cls, T5ForConditionalGeneration=mock_model_cls)

    mock_torch = MagicMock()
    mock_torch.no_grad = lambda: MagicMock()

    with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": mock_torch}):
        decoder = DiacNetDecoder()
        result = decoder.decode("se eranko naa si gbo o?", lang="yo")
        assert result == "ṣé ẹranko náà sì gbọ́ ọ?"
        # Language tag should be prefixed onto the input text
        called_text = mock_tokenizer.call_args[0][0]
        assert called_text.startswith("<yor>")


def test_diacritizer_diacnet_1_0_routes_to_decoder():
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": [1, 2, 3]}
    mock_tokenizer.decode.return_value = "c'est fini"
    mock_model = MagicMock()
    mock_model.generate.return_value = [[1, 2, 3]]

    mock_tokenizer_cls = MagicMock(from_pretrained=MagicMock(return_value=mock_tokenizer))
    mock_model_cls = MagicMock(from_pretrained=MagicMock(return_value=mock_model))
    mock_transformers = MagicMock(AutoTokenizer=mock_tokenizer_cls, T5ForConditionalGeneration=mock_model_cls)
    mock_torch = MagicMock()
    mock_torch.no_grad = lambda: MagicMock()

    with patch.dict("sys.modules", {"transformers": mock_transformers, "torch": mock_torch}):
        diacritizer = Diacritizer(model="diacnet-1.0", lang="fr")
        result = diacritizer.restore("cest fini")
        assert result == "c'est fini"


# =========================================================================== #
# Tokenizer — otk-bpe multilingual family
# =========================================================================== #

def test_tokenizer_multilingual_variant_resolves_correct_repo(tmp_path):
    fake_file = tmp_path / "sw-150k" / "tokenizer.json"
    fake_file.parent.mkdir(parents=True)
    fake_file.write_text("{}", encoding="utf-8")

    captured = {}

    def fake_get_model_path(filename, repo_id=None):
        captured["filename"] = filename
        captured["repo_id"] = repo_id
        return str(fake_file)

    with patch("olaverse.utils.downloader.get_model_path", side_effect=fake_get_model_path), \
         patch("tokenizers.Tokenizer.from_file", return_value=MagicMock()):
        Tokenizer(lang="sw-150k")

    assert captured["filename"] == "sw-150k/tokenizer.json"
    assert captured["repo_id"] == "olaverse/otk-bpe"


def test_tokenizer_nigerian_variant_still_uses_otk_bpe_50k_repo(tmp_path):
    fake_file = tmp_path / "otk-bpe-50k-yo.json"
    fake_file.write_text("{}", encoding="utf-8")

    captured = {}

    def fake_get_model_path(filename, repo_id=None):
        captured["filename"] = filename
        captured["repo_id"] = repo_id
        return str(fake_file)

    with patch("olaverse.utils.downloader.get_model_path", side_effect=fake_get_model_path), \
         patch("tokenizers.Tokenizer.from_file", return_value=MagicMock()):
        Tokenizer(lang="yoruba")

    assert captured["filename"] == "otk-bpe-50k-yo.json"
    assert captured["repo_id"] == "olaverse/otk-bpe-50k"


# =========================================================================== #
# Reranker / Embedder
# =========================================================================== #

def test_reranker_alias_resolution():
    assert Reranker(size="150m").model_name == "olaverse/mist-reranker-150m"
    assert Reranker(size="22.7m").model_name == "olaverse/mist-reranker-22.7M"


def test_reranker_import_error():
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        reranker = Reranker()
        with pytest.raises(ImportError) as exc_info:
            reranker.load()
        assert "sentence-transformers" in str(exc_info.value)


def test_reranker_rank_mocked_2class_logits():
    # raw.dim() == 2 -> 2-class logits path (e.g. mist-reranker-22.7M):
    # scores = torch.softmax(raw, dim=-1)[:, 1]
    mock_raw = MagicMock()
    mock_raw.dim.return_value = 2

    mock_scores = MagicMock()
    mock_scores.tolist.return_value = [0.9, 0.2]
    mock_softmax_result = MagicMock()
    mock_softmax_result.__getitem__.return_value = mock_scores

    mock_torch = MagicMock()
    mock_torch.softmax.return_value = mock_softmax_result

    mock_ce_instance = MagicMock()
    mock_ce_instance.predict.return_value = mock_raw
    mock_cross_encoder_cls = MagicMock(return_value=mock_ce_instance)
    mock_st = MagicMock(CrossEncoder=mock_cross_encoder_cls)

    with patch.dict("sys.modules", {"sentence_transformers": mock_st, "torch": mock_torch}):
        reranker = Reranker(size="22.7m")
        ranked = reranker.rank("q", ["relevant passage", "irrelevant passage"])

    assert ranked[0][0] == 0
    assert ranked[0][1] == 0.9
    assert ranked[1][1] == 0.2


def test_embedder_import_error():
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        embedder = Embedder()
        with pytest.raises(ImportError) as exc_info:
            embedder.load()
        assert "sentence-transformers" in str(exc_info.value)


def test_embedder_similarity():
    embedder = Embedder()
    assert embedder.similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert embedder.similarity([1, 0], [0, 1]) == pytest.approx(0.0)


# =========================================================================== #
# Vision — PrismUpscaler / PrismSteganography
# =========================================================================== #

def test_prism_upscaler_invalid_size():
    with pytest.raises(ValueError):
        PrismUpscaler(size="8x")


def test_prism_upscaler_repo_mapping():
    assert PrismUpscaler(size="2x").repo_id == "olaverse/prism-upscaler-2x"
    assert PrismUpscaler(size="4x").repo_id == "olaverse/prism-upscaler-4x"
    assert PrismUpscaler(size="max").repo_id == "olaverse/prism-upscaler-max"


def test_prism_upscaler_import_error():
    with patch.dict("sys.modules", {"torch": None}):
        upscaler = PrismUpscaler(size="2x")
        with pytest.raises(ImportError) as exc_info:
            upscaler.load()
        assert "torch" in str(exc_info.value)


def test_prism_denoiser_repo_id():
    assert PrismDenoiser().repo_id == "olaverse/prism-denoiser"


def test_prism_denoiser_import_error():
    with patch.dict("sys.modules", {"torch": None}):
        denoiser = PrismDenoiser()
        with pytest.raises(ImportError) as exc_info:
            denoiser.load()
        assert "torch" in str(exc_info.value)


def test_prism_steganography_import_error():
    with patch.dict("sys.modules", {"torch": None}):
        steg = PrismSteganography()
        with pytest.raises(ImportError) as exc_info:
            steg.load()
        assert "torch" in str(exc_info.value)


def test_prism_steganography_repo_id():
    assert PrismSteganography().repo_id == "olaverse/prism-steganography"


# =========================================================================== #
# Downloader — cache must be namespaced per repo_id (regression: every
# olaverse/prism-* repo ships identically-named model.py/config.json/
# pytorch_model.pt files; a flat cache keyed only on filename would return
# one Prism model's files when asked for a different one).
# =========================================================================== #

def test_get_model_path_cache_does_not_collide_across_repos(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("OLAVERSE_MODELS_DIR", raising=False)

    responses = {
        "olaverse/repo-a": b"class A: pass",
        "olaverse/repo-b": b"class B: pass",
    }

    def fake_urlopen(req, context=None):
        url = req.full_url
        for repo_id, body in responses.items():
            if repo_id in url:
                mock_response = MagicMock()
                mock_response.read.return_value = body
                mock_response.__enter__.return_value = mock_response
                mock_response.__exit__.return_value = False
                return mock_response
        raise AssertionError(f"unexpected URL: {url}")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        path_a = get_model_path("model.py", repo_id="olaverse/repo-a")
        path_b = get_model_path("model.py", repo_id="olaverse/repo-b")

    assert path_a != path_b
    with open(path_a, "rb") as f:
        assert f.read() == b"class A: pass"
    with open(path_b, "rb") as f:
        assert f.read() == b"class B: pass"
