import librosa
import numpy as np

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
    if progress_bar:
        progress_bar.setValue(60)

    # Timbre features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)
    if progress_bar:
        progress_bar.setValue(80)

    # Harmonic features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_means = chroma.mean(axis=1)

    features = np.concatenate([
        [spectral_centroid, spectral_bandwidth, np.mean(spectral_contrast)],
        mfcc_means,
        chroma_means
    ])

    basic_stats = {
        "Tempo": f"{tempo:.2f} BPM",
        "Spectral Centroid": f"{spectral_centroid:.2f} Hz",
        "Spectral Bandwidth": f"{spectral_bandwidth:.2f} Hz"
    }

    return features, basic_stats