from abc import ABC, abstractmethod

class BaseAcousticModel(ABC):
    """
    Abstract base class for Acoustic Models (e.g., FastSpeech, Tacotron).
    These models convert normalized/diacritized phonetic text into acoustic features like Mel-spectrograms.
    """
    @abstractmethod
    def load_weights(self, path: str):
        """Load PyTorch/ONNX model weights from the specified path."""
        pass
        
    @abstractmethod
    def forward(self, text: str):
        """
        Convert text into acoustic features.
        
        Args:
            text: Phonetically normalized text.
        Returns:
            Acoustic features (e.g., a Mel-spectrogram tensor).
        """
        pass

class BaseVocoder(ABC):
    """
    Abstract base class for Vocoders (e.g., HiFi-GAN, WaveGlow).
    These models convert acoustic features (like Mel-spectrograms) into raw audio waveforms.
    """
    @abstractmethod
    def load_weights(self, path: str):
        """Load PyTorch/ONNX model weights from the specified path."""
        pass
        
    @abstractmethod
    def generate(self, acoustic_features):
        """
        Convert acoustic features into a raw audio waveform.
        
        Args:
            acoustic_features: Output from an Acoustic Model.
        Returns:
            Audio waveform array/tensor.
        """
        pass
