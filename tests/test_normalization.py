import pytest
from olaverse.nlp.normalization import TTSNormalizer

def test_tts_normalizer_abbreviations():
    normalizer = TTSNormalizer(lang="yo")
    
    assert normalizer.expand_abbreviations("Mr. Ojo") == "Míṣìtà Ojo"
    assert normalizer.expand_abbreviations("Hello Dr. Tunde") == "Hello Dọ́kítà Tunde"
    assert normalizer.expand_abbreviations("No abbreviations here") == "No abbreviations here"

def test_tts_normalizer_numbers():
    normalizer = TTSNormalizer(lang="yo")
    
    # 123 -> ọ̀kan éjì ẹ́ta
    assert normalizer.expand_numbers("123") == "ọ̀kan éjì ẹ́ta"
    assert normalizer.expand_numbers("Call 911") == "Call ẹ́sàn-án ọ̀kan ọ̀kan"
    
def test_tts_normalizer_full_pipeline():
    normalizer = TTSNormalizer(lang="ig")
    
    assert normalizer.normalize("Mr. Obi has 2 cars") == "Mista Obi has abụo cars"
