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

def prepare_fma_dataset(audio_dir, metadata_path, output_path, num_files=None, progress_callback=None):
    tracks = load_metadata(metadata_path)
    
    if num_files is not None:
        tracks = tracks.head(num_files)
    
    total_files = len(tracks)
    features = []
    genres = []
    
    for index, (track_id, row) in enumerate(tracks.iterrows()):
        file_path = get_audio_path(audio_dir, track_id)
        if os.path.exists(file_path):
            try:
                feature = extract_features(file_path)
                features.append(feature)
                genres.append(row['track', 'genre_top'])
                print(f"Successfully processed {file_path} ({index+1}/{total_files})")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File does not exist: {file_path}")

        if progress_callback:
            progress = int((index + 1) / total_files * 100)
            progress_callback(progress)

    X = np.array(features)
    y = np.array(genres)
    
    np.savez(output_path, X=X, y=y)
    print(f"Dataset saved to {output_path}")
    print(f"Total tracks processed: {len(X)}")   
   
if __name__ == "__main__":
    audio_dir = r"F:\Audio Data Sets\FMA\fma_small"
    metadata_path = r"F:\Audio Data Sets\FMA\fma_metadata\tracks.csv"
    output_path = "F:\Audio Data Sets\FMA\fma_small_features.npz"
    num_files = 1000  # Process only 1000 files
    prepare_fma_dataset(audio_dir, metadata_path, output_path, num_files)