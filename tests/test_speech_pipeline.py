import pytest
from olaverse.speech.pipeline import TTSPipeline
from olaverse.speech.base import BaseAcousticModel, BaseVocoder

# Create mock models for testing
class MockAcousticModel(BaseAcousticModel):
    def load_weights(self, path): pass
    def forward(self, text): return "mock_mel_spectrogram"

class MockVocoder(BaseVocoder):
    def load_weights(self, path): pass
    def generate(self, acoustic_features): return "mock_waveform"

def test_tts_pipeline_text_only():
    # Test pipeline without acoustic models (returns processed text)
    pipeline = TTSPipeline(lang="yo", diacritizer_model="diacnet-yor-viterbi")
    
    result = pipeline.synthesize("Mr. Ojo lo si oja lana")
    
    assert result["status"] == "Acoustic model or Vocoder not provided."
    assert result["normalized_text"] == "Míṣìtà Ojo lo si oja lana"
    assert result["diacritized_text"] == "Míṣìtà Òjó lọ sí ọjà lana"
    assert result["audio"] is None

def test_tts_pipeline_full_synthesis():
    # Test pipeline with mock models
    pipeline = TTSPipeline(
        lang="yo", 
        acoustic_model=MockAcousticModel(), 
        vocoder=MockVocoder(),
        diacritizer_model="diacnet-yor-viterbi"
    )
    
    result = pipeline.synthesize("Mr. Ojo lo si oja lana")
    
    assert result["status"] == "Success"
    assert result["diacritized_text"] == "Míṣìtà Òjó lọ sí ọjà lana"
    assert result["audio"] == "mock_waveform"
