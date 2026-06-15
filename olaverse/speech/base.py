import warnings
from abc import ABC, abstractmethod


class ExperimentalWarning(UserWarning):
    """
    Raised when using olaverse.speech classes that have no trained model behind them yet.
    The diacritizers and TTS normalizer are production-ready; acoustic synthesis is on the roadmap.

    To silence this warning:
        import warnings
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
    """
    pass


class BaseAcousticModel(ABC):
    """
    Abstract base class for acoustic models (e.g. FastSpeech, Tacotron).
    Converts normalised/diacritised phonetic text into Mel-spectrograms.

    .. warning:: **Experimental — no trained model available yet.**
        Subclassing this is fine for custom integrations, but olaverse does not
        yet ship a trained acoustic model. This is on the roadmap.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from BaseAcousticModel which is part of the "
            "experimental olaverse.speech module. No trained acoustic model is currently "
            "available from olaverse — you will need to supply your own weights. "
            "This feature is on the roadmap.",
            ExperimentalWarning,
            stacklevel=2,
        )

    @abstractmethod
    def load_weights(self, path: str):
        """Load PyTorch/ONNX model weights from the specified path."""
        pass

    @abstractmethod
    def forward(self, text: str):
        """
        Convert text into acoustic features (e.g. a Mel-spectrogram tensor).

        Args:
            text: Phonetically normalised and diacritised text.
        Returns:
            Acoustic features tensor.
        """
        pass


class BaseVocoder(ABC):
    """
    Abstract base class for vocoders (e.g. HiFi-GAN, WaveGlow).
    Converts Mel-spectrograms into raw audio waveforms.

    .. warning:: **Experimental — no trained model available yet.**
        Subclassing this is fine for custom integrations, but olaverse does not
        yet ship a trained vocoder. This is on the roadmap.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        warnings.warn(
            f"{cls.__name__} inherits from BaseVocoder which is part of the "
            "experimental olaverse.speech module. No trained vocoder is currently "
            "available from olaverse — you will need to supply your own weights. "
            "This feature is on the roadmap.",
            ExperimentalWarning,
            stacklevel=2,
        )

    @abstractmethod
    def load_weights(self, path: str):
        """Load PyTorch/ONNX model weights from the specified path."""
        pass

    @abstractmethod
    def generate(self, acoustic_features: object) -> object:
        """
        Convert acoustic features into a raw audio waveform.

        Args:
            acoustic_features: Output from a BaseAcousticModel.
        Returns:
            Audio waveform array/tensor.
        """
        pass
