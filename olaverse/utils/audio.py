import os

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def save_audio(waveform, sample_rate: int, output_path: str):
    """
    Save a generated audio waveform to a .wav file.
    
    Args:
        waveform: 1D numpy array representing the audio signal.
        sample_rate: The sampling rate (e.g., 22050 or 24000).
        output_path: Path to save the audio file.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required to save audio files. Install it via `pip install scipy`.")
        
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wavfile.write(output_path, sample_rate, waveform)

def load_audio(input_path: str):
    """
    Load an audio waveform from a .wav file.
    
    Returns:
        tuple: (sample_rate, waveform)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required to load audio files. Install it via `pip install scipy`.")
        
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Audio file not found at: {input_path}")
        
    return wavfile.read(input_path)
