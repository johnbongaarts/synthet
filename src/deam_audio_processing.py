import librosa
import numpy as np

def analyze_audio_deam(file_path):
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, duration=45)  # DEAM uses 45-second excerpts

        # Extract features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = chroma.mean(axis=1)
        rms = librosa.feature.rms(y=y).mean()

        # Ensure all spectral features are scalar values
        spectral_features = np.array([
            float(spectral_centroid),
            float(spectral_bandwidth),
            float(spectral_rolloff),
            float(tempo),
            float(rms)
        ])
        
        # Concatenate all features
        features = np.concatenate([
            spectral_features,
            mfcc_means.flatten(),
            chroma_means.flatten()
        ])

        # Print shapes for debugging
        print(f"Spectral features shape: {spectral_features.shape}")
        print(f"MFCC means shape: {mfcc_means.shape}")
        print(f"Chroma means shape: {chroma_means.shape}")
        print(f"Final features shape: {features.shape}")

        return features

    except Exception as e:
        print(f"Error in analyze_audio_deam for {file_path}: {str(e)}")
        return None

def get_feature_names():
    return [
        "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
        "tempo", "rms",
        *[f"mfcc_{i}" for i in range(13)],
        *[f"chroma_{i}" for i in range(12)]
    ]

def get_feature_count():
    return len(get_feature_names())

def debug_feature_extraction(file_path):
    try:
        y, sr = librosa.load(file_path, duration=45)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = chroma.mean(axis=1)
        rms = librosa.feature.rms(y=y).mean()

        print(f"spectral_centroid: {spectral_centroid}, shape: {np.array(spectral_centroid).shape}, type: {type(spectral_centroid)}")
        print(f"spectral_bandwidth: {spectral_bandwidth}, shape: {np.array(spectral_bandwidth).shape}, type: {type(spectral_bandwidth)}")
        print(f"spectral_rolloff: {spectral_rolloff}, shape: {np.array(spectral_rolloff).shape}, type: {type(spectral_rolloff)}")
        print(f"tempo: {tempo}, shape: {np.array(tempo).shape}, type: {type(tempo)}")
        print(f"mfcc_means shape: {mfcc_means.shape}, type: {type(mfcc_means)}")
        print(f"chroma_means shape: {chroma_means.shape}, type: {type(chroma_means)}")
        print(f"rms: {rms}, shape: {np.array(rms).shape}, type: {type(rms)}")

        # Print the shape of the final feature vector
        features = analyze_audio_deam(file_path)
        if features is not None:
            print(f"Final feature vector shape: {features.shape}")
            print(f"Expected feature count: {get_feature_count()}")
            if features.shape[0] != get_feature_count():
                print("WARNING: Feature count mismatch!")
        else:
            print("Failed to extract features")

    except Exception as e:
        print(f"Error in debug_feature_extraction for {file_path}: {str(e)}")