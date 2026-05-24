import pytest
import os
import json
import shutil
from unittest.mock import patch, MagicMock
from olaverse.utils.constants import STATES, BANKS, format_naira, get_telco
from olaverse.nlp.preprocessing import mask_pii, is_pidgin_particle
from olaverse.nlp.language_detection import detect_language
from olaverse.nlp.diacritizer import diacritize_yoruba, diacritize_yoruba_dot_below, diacritize_igbo
from olaverse.nlp.sentiment import analyze_sentiment
from olaverse.nlp.tokenizer import Tokenizer
from olaverse.data.loaders import load_dataset
from olaverse.utils.downloader import get_model_path, get_cache_dir

def test_constants():
    # States and Capitals
    assert STATES["Lagos"] == "Ikeja"
    assert STATES["Oyo"] == "Ibadan"
    
    # Banks and Codes
    assert BANKS["Guaranty Trust Bank"] == "058"
    assert BANKS["Zenith Bank"] == "057"
    
    # Naira formatter
    assert format_naira(1500000) == "₦1,500,000.00"
    assert format_naira(500) == "₦500.00"
    
    # Telco prefixes
    assert get_telco("08031234567") == "MTN"
    assert get_telco("+2348021234567") == "Airtel"
    assert get_telco("08051111111") == "Glo"
    assert get_telco("08092222222") == "9mobile"
    assert get_telco("12345") is None

def test_preprocessing():
    # PII masking
    masked_phone = mask_pii("Call me on 08012345678 or email me@example.com")
    assert "[PHONE]" in masked_phone
    assert "[EMAIL]" in masked_phone
    
    masked_bvn = mask_pii("My BVN is 22233344455")
    assert "[BVN]" in masked_bvn
    
    # Pidgin particles
    assert is_pidgin_particle("sha") is True
    assert is_pidgin_particle("sef") is True
    assert is_pidgin_particle("abeg") is True
    assert is_pidgin_particle("table") is False

def test_language_detection():
    # Detect typical phrases
    assert detect_language("Bawo ni, se daadaa ni?") == "yor"
    assert detect_language("Ina kwana? Lafiya lau.") == "hau"
    assert detect_language("Kedu ka ị mere?") == "ibo"
    assert detect_language("How far, wetin dey happen?") == "pcm"
    assert detect_language("How are you doing today?") == "eng"

def test_diacritizers():
    # Yoruba diacritics
    # "Ojo lo si oja lana" -> "Ọjọ́ ló sí ọjà lànà"
    assert diacritize_yoruba("Ojo lo si oja lana") == "Ọjọ́ ló sí ọjà lànà"
    
    # Dot-below only
    assert diacritize_yoruba_dot_below("Ojo lo si oja lana") == "Ọjọ lo si ọja lana"
    
    # Igbo diacritics
    assert diacritize_igbo("Kedu ka i mere") == "Kedụ ka ị mere"

def test_sentiment_analysis():
    # Positive sentiment
    pos_res = analyze_sentiment("This film too sweet!")
    assert pos_res["label"] == "positive"
    assert pos_res["confidence"] > 0.5
    
    # Negative sentiment
    neg_res = analyze_sentiment("I no like am at all")
    assert neg_res["label"] == "negative"
    assert neg_res["confidence"] > 0.5

def test_tokenizers():
    # Test tokenizer creation and encode/decode roundtrip
    tok = Tokenizer("naija")
    
    input_text = "Ẹ kú àbọ̀"
    ids = tok.encode(input_text)
    decoded = tok.decode(ids)
    
    assert len(ids) > 0
    # Decoded should match input (diacritic preservation)
    assert decoded.strip() == input_text.strip()

    # Test direct model name loading without .json extension
    tok_direct = Tokenizer("otk-bpe-50k-yo")
    ids_direct = tok_direct.encode(input_text)
    assert len(ids_direct) > 0
    assert tok_direct.decode(ids_direct).strip() == input_text.strip()

def test_dataset_loaders():
    # Load sample data
    data_ns = load_dataset("naijasenti", lang="yor", split="train")
    assert len(data_ns) > 0
    assert "text" in data_ns[0]
    assert "label" in data_ns[0]
    
    data_ner = load_dataset("masakhaner", lang="hau", split="train")
    assert len(data_ner) > 0
    assert "tokens" in data_ner[0]
    assert "ner_tags" in data_ner[0]

def test_downloader_custom_dir(tmp_path):
    # Set up custom models directory via environment variable
    custom_dir = tmp_path / "custom_models"
    custom_dir.mkdir()
    model_file = custom_dir / "test_model.json"
    model_file.write_text('{"key": "value"}', encoding="utf-8")
    
    with patch.dict(os.environ, {"OLAVERSE_MODELS_DIR": str(custom_dir)}):
        resolved_path = get_model_path("test_model.json")
        assert resolved_path == str(model_file)

def test_downloader_cache_dir(tmp_path):
    # Set up custom cache directory via XDG_CACHE_HOME env variable
    custom_cache = tmp_path / "cache"
    custom_cache.mkdir()
    
    model_dir = custom_cache / "olaverse" / "models"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "test_cache_model.json"
    model_file.write_text('{"cache_key": "cache_value"}', encoding="utf-8")
    
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(custom_cache)}):
        # Clean environment to make sure OLAVERSE_MODELS_DIR is not set
        if "OLAVERSE_MODELS_DIR" in os.environ:
            del os.environ["OLAVERSE_MODELS_DIR"]
        resolved_path = get_model_path("test_cache_model.json")
        assert resolved_path == str(model_file)

def test_downloader_huggingface_download(tmp_path):
    # Test downloader fetching from HF when local cache, custom, and pkg directories do not contain it
    custom_cache = tmp_path / "cache"
    custom_cache.mkdir()
    
    model_dir = custom_cache / "olaverse" / "models"
    # Ensure cache directory is empty/non-existent initially
    
    # We mock urllib.request.urlretrieve to simulate a download
    def mock_urlretrieve(url, filename, reporthook=None, data=None):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write('{"downloaded": true}')
            
    with patch("urllib.request.urlretrieve", side_effect=mock_urlretrieve) as mock_retrieve, \
         patch.dict(os.environ, {"XDG_CACHE_HOME": str(custom_cache)}):
         
        # Ensure clean environment
        if "OLAVERSE_MODELS_DIR" in os.environ:
            del os.environ["OLAVERSE_MODELS_DIR"]
            
        resolved_path = get_model_path("downloaded_model.json")
        expected_path = os.path.join(str(model_dir), "downloaded_model.json")
        assert resolved_path == expected_path
        
        # Check that urlretrieve was called with the correct Hugging Face URL
        expected_url = "https://huggingface.co/olaverse/otk-bpe-50k/resolve/main/downloaded_model.json"
        mock_retrieve.assert_called_once()
        args, kwargs = mock_retrieve.call_args
        assert args[0] == expected_url
        assert args[1] == expected_path
        
        # Verify content was written
        with open(resolved_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["downloaded"] is True

def test_downloader_huggingface_download_failure(tmp_path):
    # Test download failure behavior when offline/error occurs
    custom_cache = tmp_path / "cache"
    custom_cache.mkdir()
    
    with patch("urllib.request.urlretrieve", side_effect=Exception("Connection timed out")), \
         patch.dict(os.environ, {"XDG_CACHE_HOME": str(custom_cache)}):
         
        # Ensure clean environment
        if "OLAVERSE_MODELS_DIR" in os.environ:
            del os.environ["OLAVERSE_MODELS_DIR"]
            
        with pytest.raises(RuntimeError) as exc_info:
            get_model_path("nonexistent_model.json")
            
        assert "Failed to download model file" in str(exc_info.value)
        assert "Connection timed out" in str(exc_info.value)

def test_language_detection_custom_path(tmp_path):
    # Test passing custom path to language detector
    model_file = tmp_path / "dummy_lang_model.json"
    dummy_model = {
        "priors": {"yor": -0.5, "eng": -0.5},
        "features": {
            "_ola_": {"yor": 2.0, "eng": -2.0}
        },
        "default_log_prob": -10.0
    }
    model_file.write_text(json.dumps(dummy_model), encoding="utf-8")
    
    # "Ola" should trigger the "_ola_" n-gram and classify as yor
    # "Hello" should not match "_ola_" and default to priors (or default_log_prob)
    res_yor = detect_language("Ola", model_path=str(model_file))
    assert res_yor == "yor"
    
    # Check that nonexistent file path raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        detect_language("Ola", model_path="nonexistent_file.json")

def test_sentiment_custom_path(tmp_path):
    # Test passing custom path to sentiment analyzer
    model_file = tmp_path / "dummy_sentiment_model.json"
    dummy_model = {
        "vocab": {"sweet": 0, "bad": 1},
        "idf": [1.0, 1.0],
        "coef": [2.0, -2.0],
        "intercept": 0.0
    }
    model_file.write_text(json.dumps(dummy_model), encoding="utf-8")
    
    res = analyze_sentiment("sweet", model_path=str(model_file))
    assert res["label"] == "positive"
    assert res["confidence"] > 0.5
    
    res_neg = analyze_sentiment("bad", model_path=str(model_file))
    assert res_neg["label"] == "negative"
    assert res_neg["confidence"] > 0.5
    
    with pytest.raises(FileNotFoundError):
        analyze_sentiment("sweet", model_path="nonexistent_file.json")

def test_diacritizer_custom_path(tmp_path):
    # Test passing custom path to Yoruba/Igbo diacritizer
    model_file = tmp_path / "dummy_diacritizer.json"
    dummy_model = {
        "candidates": {
            "ojo": ["ọjọ́"]
        },
        "transitions": {},
        "unigrams": {
            "ọjọ́": 0.0
        }
    }
    model_file.write_text(json.dumps(dummy_model), encoding="utf-8")
    
    res = diacritize_yoruba("Ojo", model_path=str(model_file))
    assert res == "Ọjọ́"
    
    with pytest.raises(FileNotFoundError):
        diacritize_yoruba("Ojo", model_path="nonexistent_file.json")

def test_tokenizer_custom_path(tmp_path):
    # Test passing custom path to Tokenizer
    # Copy an existing tokenizer file to test custom path loading
    pkg_models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "olaverse", "models")
    src_tokenizer = os.path.join(pkg_models_dir, "tokenizer_hausa.json")
    
    if os.path.exists(src_tokenizer):
        dest_tokenizer = tmp_path / "custom_tokenizer_hausa.json"
        shutil.copy(src_tokenizer, dest_tokenizer)
        
        tok = Tokenizer(lang="hausa", model_path=str(dest_tokenizer))
        ids = tok.encode("Ina kwana")
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert decoded.strip() == "Ina kwana"
        
    with pytest.raises(FileNotFoundError):
        Tokenizer(lang="hausa", model_path="nonexistent_tokenizer.json")
