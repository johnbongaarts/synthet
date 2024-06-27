import librosa
import numpy as np
from src.feature_config import get_feature_config

FEATURE_CONFIG = get_feature_config()

# Define feature indices
TOTAL_FEATURES = 28
TEMPO_INDEX = 0
SPECTRAL_CENTROID_INDEX = 1
SPECTRAL_BANDWIDTH_INDEX = 2
SPECTRAL_CONTRAST_START = 3
SPECTRAL_CONTRAST_END = 9  # Assuming 6 spectral contrast values
SPECTRAL_ROLLOFF_INDEX = 9
CHROMA_START = 10
CHROMA_END = 15  # 5 chroma features instead of 12
MFCC_START = 15
MFCC_END = 27  # Assuming 12 MFCC features
RMS_INDEX = 27

# Verification
assert RMS_INDEX + 1 == TOTAL_FEATURES, "Feature indices don't match total feature count"

def analyze_audio(file_path, progress_bar=None):
    y, sr = librosa.load(file_path)
    if progress_bar:
        progress_bar.setValue(20)

    # Basic features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = tempo.item() if isinstance(tempo, np.ndarray) else tempo  # Convert to scalar if it's an array
    if progress_bar:
        progress_bar.setValue(40)

    # Spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    if progress_bar:
        progress_bar.setValue(60)

    # Timbre features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
    mfcc_means = mfccs.mean(axis=1)
    

    # Harmonic features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_means = chroma.mean(axis=1)

    # Energy
    rms = librosa.feature.rms(y=y).mean()

    # Combine all features
    features = np.zeros(FEATURE_CONFIG['TOTAL_FEATURES'])
    features[FEATURE_CONFIG['TEMPO']] = tempo
    features[FEATURE_CONFIG['SPECTRAL_CENTROID']] = spectral_centroid
    features[FEATURE_CONFIG['SPECTRAL_BANDWIDTH']] = spectral_bandwidth
    features[FEATURE_CONFIG['SPECTRAL_CONTRAST_START']:FEATURE_CONFIG['SPECTRAL_CONTRAST_END']] = spectral_contrast[:6]
    features[FEATURE_CONFIG['SPECTRAL_ROLLOFF']] = spectral_rolloff
    features[FEATURE_CONFIG['CHROMA_START']:FEATURE_CONFIG['CHROMA_END']] = chroma_means[:5]
    features[FEATURE_CONFIG['MFCC_START']:FEATURE_CONFIG['MFCC_END']] = mfcc_means
    features[FEATURE_CONFIG['RMS']] = rms
    
    if progress_bar:
        progress_bar.setValue(80)

    basic_stats = {
        "Tempo": f"{tempo:.2f} BPM",
        "Spectral Centroid": f"{spectral_centroid:.2f} Hz",
        "Spectral Bandwidth": f"{spectral_bandwidth:.2f} Hz",
        "RMS Energy": f"{rms:.4f}"
    }

    return features, basic_stats

# At the end of src/audio_processing.py
__all__ = ['analyze_audio', 'TOTAL_FEATURES']