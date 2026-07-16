"""
Real end-to-end tests — no mocks.

Every test here downloads the actual model or dataset from Hugging Face and
asserts on real outputs (all expected values were verified against live runs).

Two tiers, controlled by pytest markers:

  * ``integration`` — light real tests (small JSON/tokenizer/Prism files, a
    few MB). Run by default; need network on first run, then hit the local
    model cache.
  * ``slow`` — full transformer checkpoints (90MB+ downloads). Excluded by
    default; run with:  pytest -m slow

Tests for optional-dependency models skip cleanly when the extra isn't
installed (e.g. fasttext for LIDLite25).
"""

import pytest

integration = pytest.mark.integration
slow = pytest.mark.slow


def _installed(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


needs_st = pytest.mark.skipif(
    not _installed("sentence_transformers"),
    reason="requires olaverse[retrieval] (sentence-transformers)",
)
needs_transformers = pytest.mark.skipif(
    not _installed("transformers"),
    reason="requires olaverse[deeplearning] (transformers/torch)",
)
needs_torch = pytest.mark.skipif(
    not (_installed("torch") and _installed("torchvision") and _installed("PIL")),
    reason="requires olaverse[vision] (torch/torchvision/Pillow)",
)
needs_fasttext = pytest.mark.skipif(
    not _installed("fasttext"),
    reason="requires olaverse[lid] (fasttext)",
)
needs_datasets = pytest.mark.skipif(
    not _installed("datasets"),
    reason="requires olaverse[data] (datasets)",
)


# =========================================================================== #
# Language detection — real lid-lite-5 model
# =========================================================================== #

@integration
def test_detect_language_real_all_five_languages():
    from olaverse import detect_language

    assert detect_language("Bawo ni o se wa loni") == "yor"
    assert detect_language("Ina kwana, yaya aiki?") == "hau"
    assert detect_language("Kedu ka i mere taa?") == "ibo"
    assert detect_language("Wetin dey happen for area na") == "pcm"
    assert detect_language("How are you doing today?") == "eng"


@integration
def test_lid_lite5_probabilities_are_a_distribution():
    from olaverse import LIDLite5

    probs = LIDLite5().predict_proba("Bawo ni o se wa loni")
    assert set(probs) == {"yor", "hau", "ibo", "pcm", "eng"}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert max(probs, key=probs.get) == "yor"


# =========================================================================== #
# Tokenizer — real otk-bpe-50k models
# =========================================================================== #

@integration
def test_tokenizer_yoruba_real_roundtrip():
    from olaverse import Tokenizer

    tok = Tokenizer("yoruba")
    text = "Bawo ni"
    ids = tok.encode(text)
    assert ids and all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == text


@integration
def test_tokenizer_naija_real_roundtrip():
    from olaverse import Tokenizer

    tok = Tokenizer("naija")
    text = "Wetin dey happen"
    assert tok.decode(tok.encode(text)) == text


# =========================================================================== #
# Diacritization — real Viterbi / KNN models
# =========================================================================== #

@integration
def test_diacritize_yoruba_real_output():
    from olaverse import diacritize_yoruba

    assert diacritize_yoruba("Ojo lo si oja lana") == "Òjó lọ sí ọjà lana"


@integration
def test_diacritize_igbo_real_output():
    from olaverse import diacritize_igbo

    assert diacritize_igbo("O na aga n ulo akwukwo") == "Ọ na aga n ụlọ akwụkwọ"


@integration
def test_diacritizer_auto_detects_and_restores():
    from olaverse.nlp import Diacritizer

    d = Diacritizer(model="auto")
    assert d.restore("Ojo lo si oja lana") == "Òjó lọ sí ọjà lana"


# =========================================================================== #
# Datasets — real Hugging Face loads
# =========================================================================== #

@integration
@needs_datasets
def test_load_diacbench_real():
    from olaverse import load_dataset

    ds = load_dataset("diacbench", "yo", split="test")
    assert len(ds) == 1000
    row = ds[0]
    assert set(row) == {"input", "reference"}
    # The reference carries diacritics the input lacks
    assert row["input"] != row["reference"]


def test_list_datasets_matches_registry():
    from olaverse import list_datasets, dataset_info

    names = list_datasets()
    assert "diacbench" in names
    assert "reranker-general-en-llm-judged" in names
    info = dataset_info("olaverse/diacbench")  # full repo ID also accepted
    assert info["repo_id"] == "olaverse/diacbench"
    assert "yo" in info["configs"]


def test_load_dataset_config_validation():
    from olaverse import load_dataset

    with pytest.raises(ValueError, match="requires a config"):
        load_dataset("diacbench")
    with pytest.raises(ValueError, match="Unknown config"):
        load_dataset("diacbench", "zz")
    with pytest.raises(ValueError, match="Unknown olaverse dataset"):
        load_dataset("not-a-dataset")


# =========================================================================== #
# Vision — real Prism models (tiny checkpoints, still light)
# =========================================================================== #

def _natural_test_image():
    """Synthetic photo-like image (smooth gradients + a soft disc).

    Prism models are trained on natural photos — pure random noise is
    out-of-distribution and steganography recovery fails on it.
    """
    from PIL import Image
    import numpy as np

    y, x = np.mgrid[0:128, 0:128]
    arr = np.stack([
        x / 128 * 180 + 40,
        y / 128 * 140 + 60,
        (x + y) / 256 * 160 + 50,
    ], axis=-1)
    arr[((x - 64) ** 2 + (y - 70) ** 2) < 900] = [200, 180, 90]
    return Image.fromarray(arr.astype("uint8"))


@integration
@needs_torch
def test_prism_upscaler_2x_real():
    from olaverse import PrismUpscaler

    img = _natural_test_image().resize((64, 64))
    out = PrismUpscaler(size="2x").upscale(img)
    assert out.size == (128, 128)


@integration
@needs_torch
def test_prism_steganography_real_roundtrip():
    from olaverse import PrismSteganography

    steg = PrismSteganography()
    stego = steg.hide(_natural_test_image(), "hi olax")
    assert steg.reveal(stego) == "hi olax"


@integration
@needs_torch
def test_prism_denoiser_real_shape():
    from olaverse import PrismDenoiser

    out = PrismDenoiser().denoise(_natural_test_image())
    assert out.size == (128, 128)


# =========================================================================== #
# LIDLite25 — real fastText checkpoints (skips without olaverse[lid])
# =========================================================================== #

@integration
@needs_fasttext
def test_lid_lite_25_real():
    from olaverse import LIDLite25

    det = LIDLite25(variant="questions")
    assert det.predict("What causes ocean tides?") == "eng"
    probs = det.predict_proba("Qu'est-ce qui cause les marées ?")
    assert max(probs, key=probs.get) == "fra"


# =========================================================================== #
# Heavy real-model tests — full transformer downloads, opt in with -m slow
# =========================================================================== #

@slow
@needs_st
def test_reranker_real_ranks_relevant_passage_first():
    from olaverse import Reranker

    ranked = Reranker(size="22.7m").rank("who wrote hamlet", [
        "The capital of France is Paris.",
        "Hamlet is a tragedy written by William Shakespeare around 1600.",
        "Tidal energy is renewable.",
    ])
    assert ranked[0][0] == 1          # Shakespeare passage first
    assert ranked[0][1] > 0.85        # verified live: ~0.91
    assert all(ranked[0][1] > s for _, s in ranked[1:])


@slow
@needs_st
def test_embedder_real_semantic_similarity():
    from olaverse import Embedder

    e = Embedder()
    # Full sentences — the model card notes training was on general-domain
    # sentence pairs; short greetings are out-of-distribution.
    a, b, c = e.encode([
        "Yaro yana cin abinci a gida.",           # ha: boy eating at home
        "Yarinya tana cin abinci a makaranta.",   # ha: girl eating at school
        "Motar tana da sauri sosai a kan hanya.", # ha: the car is very fast
    ])
    assert a.shape == (384,)
    assert e.similarity(a, b) > e.similarity(a, c)


@slow
@needs_st
def test_embedder_real_cross_lingual():
    from olaverse import Embedder

    e = Embedder()
    ha_rain, yo_rain, ha_tax = e.encode([
        "Ruwan sama ya fadi jiya da yamma.",  # ha: rain fell yesterday evening
        "Ojo rọ̀ ní ìrọ̀lẹ́ àná.",              # yo: rain fell yesterday evening
        "Gwamnati za ta kara harajin mota.",   # ha: government raising car tax
    ])
    assert e.similarity(ha_rain, yo_rain) > e.similarity(ha_rain, ha_tax)


@slow
@needs_transformers
def test_lid_neural_5_1_real():
    from olaverse import LIDNeural5_1

    det = LIDNeural5_1()
    det.load()
    assert set(det.classes) == {"Hausa", "Yoruba", "Igbo", "Nigerian Pidgin"}
    assert det.predict("Ina kwana, yaya aiki?") == "Hausa"


@slow
@needs_transformers
def test_diacnet_1_0_real():
    from olaverse.nlp import Diacritizer

    d = Diacritizer(model="diacnet-1.0", lang="yo")
    out = d.restore("se eranko naa si gbo o?")
    assert out != "se eranko naa si gbo o?"
    # Output must contain restored combining diacritics
    import unicodedata
    assert any(unicodedata.category(c) == "Mn" for c in unicodedata.normalize("NFD", out))
