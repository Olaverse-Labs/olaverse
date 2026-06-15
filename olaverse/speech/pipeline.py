import warnings

from olaverse.nlp.normalization import TTSNormalizer
from olaverse.nlp.diacritizer import Diacritizer
from olaverse.speech.base import BaseAcousticModel, BaseVocoder, ExperimentalWarning


class TTSPipeline:
    """
    End-to-end Text-to-Speech pipeline architecture.

    Orchestrates: Text Normalisation → Tone Restoration → Acoustic Model → Vocoder.

    .. warning:: **Experimental — no trained acoustic model or vocoder available yet.**
        Steps 1 (normalisation) and 2 (diacritisation) are fully functional.
        Steps 3 and 4 require you to inject your own acoustic model and vocoder.
        End-to-end audio synthesis from olaverse is on the roadmap.

        To silence this warning:
            import warnings
            warnings.filterwarnings("ignore", category=ExperimentalWarning)
    """

    def __init__(
        self,
        lang: str = "yo",
        acoustic_model: BaseAcousticModel = None,
        vocoder: BaseVocoder = None,
        diacritizer_model: str = "diacnet-yor-viterbi",
    ):
        warnings.warn(
            "TTSPipeline is part of the experimental olaverse.speech module. "
            "The normalisation and diacritisation steps are production-ready, but "
            "olaverse does not yet ship a trained acoustic model or vocoder. "
            "End-to-end audio synthesis requires injecting your own models. "
            "This feature is on the roadmap.",
            ExperimentalWarning,
            stacklevel=2,
        )

        self.lang = lang
        self.normalizer = TTSNormalizer(lang=self.lang)
        self.diacritizer = Diacritizer(model=diacritizer_model)
        self.acoustic_model = acoustic_model
        self.vocoder = vocoder

    def synthesize(self, text: str):
        """
        Synthesise raw text into an audio waveform.

        Returns a dict with keys:
            - normalized_text: text after abbreviation/number expansion
            - diacritized_text: text after tone restoration
            - audio: waveform array/tensor, or None if no acoustic model/vocoder provided
            - status: "Success" or a message explaining what is missing
        """
        normalized_text = self.normalizer.normalize(text)
        diacritized_text = self.diacritizer.restore(normalized_text)

        if not self.acoustic_model or not self.vocoder:
            return {
                "normalized_text": normalized_text,
                "diacritized_text": diacritized_text,
                "audio": None,
                "status": "Acoustic model or Vocoder not provided.",
            }

        mel_spectrogram = self.acoustic_model.forward(diacritized_text)
        audio_waveform = self.vocoder.generate(mel_spectrogram)

        return {
            "normalized_text": normalized_text,
            "diacritized_text": diacritized_text,
            "audio": audio_waveform,
            "status": "Success",
        }
