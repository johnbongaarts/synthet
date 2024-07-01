import os
import pandas as pd
import numpy as np
from src.audio_processing import analyze_audio

def prepare_deam_dataset(audio_dir, annotations_file, output_file):
    annotations = pd.read_csv(annotations_file)
    features = []
    labels = []

    for _, row in annotations.iterrows():
        file_path = os.path.join(audio_dir, row['song_id'] + '.mp3')
        if os.path.exists(file_path):
            audio_features, _ = analyze_audio(file_path)
            features.append(audio_features)
            labels.append([row['valence_mean'], row['arousal_mean']])

    X = np.array(features)
    y = np.array(labels)
    
    np.savez(output_file, X=X, y=y)
    print(f"DEAM dataset prepared and saved to {output_file}")

if __name__ == "__main__":
    audio_dir = "path/to/DEAM/audio/files"
    annotations_file = "path/to/DEAM/annotations.csv"
    output_file = "deam_features.npz"
    prepare_deam_dataset(audio_dir, annotations_file, output_file)