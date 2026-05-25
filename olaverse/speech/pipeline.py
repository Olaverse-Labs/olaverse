from olaverse.nlp.normalization import TTSNormalizer
from olaverse.nlp.diacritizer import Diacritizer
from olaverse.speech.base import BaseAcousticModel, BaseVocoder

class TTSPipeline:
    """
    End-to-end Text-to-Speech pipeline.
    
    Orchestrates the entire flow:
    1. Text Normalization (Numbers, abbreviations)
    2. Tone Restoration (Diacritization)
    3. Acoustic Modeling (Text -> Mel-spectrogram)
    4. Vocoding (Mel-spectrogram -> Audio Waveform)
    """
    def __init__(self, 
                 lang: str = "yo", 
                 acoustic_model: BaseAcousticModel = None, 
                 vocoder: BaseVocoder = None,
                 diacritizer_model: str = "diacnet-yor-viterbi"):
        
        self.lang = lang
        self.normalizer = TTSNormalizer(lang=self.lang)
        
        # We integrate the unified Diacritizer registry seamlessly
        self.diacritizer = Diacritizer(model=diacritizer_model)
        
        self.acoustic_model = acoustic_model
        self.vocoder = vocoder

    def synthesize(self, text: str):
        """
        Synthesize raw text into an audio waveform.
        """
        # Step 1: Normalize (Expand "Mr.", "123", etc.)
        normalized_text = self.normalizer.normalize(text)
        
        # Step 2: Restore tones/diacritics if needed
        diacritized_text = self.diacritizer.restore(normalized_text)
        
        # Return early if models aren't injected yet (useful for testing the NLP pipeline alone)
        if not self.acoustic_model or not self.vocoder:
            return {
                "normalized_text": normalized_text,
                "diacritized_text": diacritized_text,
                "audio": None,
                "status": "Acoustic model or Vocoder not provided."
            }
            
        # Step 3: Acoustic Modeling
        mel_spectrogram = self.acoustic_model.forward(diacritized_text)
        
        # Step 4: Vocoding
        audio_waveform = self.vocoder.generate(mel_spectrogram)
        
        return {
            "normalized_text": normalized_text,
            "diacritized_text": diacritized_text,
            "audio": audio_waveform,
            "status": "Success"
        }
