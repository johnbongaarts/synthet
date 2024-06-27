import os
import pandas as pd
import numpy as np
import librosa

def load_metadata(metadata_path):
    print(f"Attempting to load metadata from {metadata_path}")
    try:
        tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
        
        # Filter for only the small dataset
        small_dataset = tracks[tracks[('set', 'subset')] == 'small']
        
        print(f"Successfully loaded {len(small_dataset)} tracks from fma_small metadata")
        return small_dataset
    except Exception as e:
        print(f"Error loading metadata: {e}")
        raise

def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_means = chroma.mean(axis=1)
    
    features = np.concatenate([
        [spectral_centroid, spectral_bandwidth, spectral_contrast],
        mfcc_means,
        chroma_means
    ])
    return features

def prepare_dataset(audio_dir, metadata_path, output_path, num_files=None):
    tracks = load_metadata(metadata_path)
    
    features = []
    genres = []
    processed_count = 0
    
    for index, row in tracks.iterrows():
        if num_files is not None and processed_count >= num_files:
            break
        
        file_path = get_audio_path(audio_dir, index)
        if os.path.exists(file_path):
            try:
                feature = extract_features(file_path)
                features.append(feature)
                genres.append(row['track', 'genre_top'])
                processed_count += 1
                print(f"Successfully processed {file_path} ({processed_count}/{num_files if num_files else len(tracks)})")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File does not exist: {file_path}")

    X = np.array(features)
    y = np.array(genres)
    
    np.savez(output_path, X=X, y=y)
    print(f"Dataset saved to {output_path}")
    print(f"Total tracks processed: {len(X)}")
    pass

__all__ = ['prepare_dataset']