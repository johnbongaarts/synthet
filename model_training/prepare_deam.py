import os
import pandas as pd
import numpy as np
from src.deam_audio_processing import analyze_audio_deam, get_feature_count, debug_feature_extraction

def load_annotations(annotations_file):
    print(f"Loading annotations from {annotations_file}")
    try:
        annotations = pd.read_csv(annotations_file)
        print(f"Successfully loaded {len(annotations)} annotations")
        print(f"Columns in the annotation file: {annotations.columns}")
        return annotations
    except Exception as e:
        print(f"Error loading annotations: {e}")
        raise

def get_audio_path(audio_dir, song_id):
    return os.path.join(audio_dir, f"{int(song_id)}.mp3")

def prepare_deam_dataset(audio_dir, annotations_file, output_file, num_files=None, progress_callback=None):
    annotations = load_annotations(annotations_file)
    
    if num_files is not None:
        annotations = annotations.head(num_files)
    
    total_files = len(annotations)
    features = []
    labels = []
    
    for index, row in annotations.iterrows():
        try:
            song_id = row['song_id']
            file_path = get_audio_path(audio_dir, song_id)
            
            if os.path.exists(file_path):
                # Debug: Print feature shapes for the first file
                if index == 0:
                    print(f"Debugging first file: {file_path}")
                    debug_feature_extraction(file_path)
                
                audio_features = analyze_audio_deam(file_path)
                if audio_features is not None:
                    if 'valence_mean' in row and 'arousal_mean' in row:
                        valence = row['valence_mean']
                        arousal = row['arousal_mean']
                        features.append(audio_features)
                        labels.append([valence, arousal])
                        print(f"Processed {file_path} ({index+1}/{total_files})")
                    else:
                        print(f"Missing valence or arousal for {file_path}. Available columns: {row.index}")
                else:
                    print(f"Failed to extract features from {file_path}")
            else:
                print(f"File does not exist: {file_path}")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            print(f"Row content: {row}")
        
        if progress_callback:
            progress = int((index + 1) / total_files * 100)
            progress_callback(progress)

    if not features:
        raise ValueError("No valid features extracted. Check your audio files and annotations.")

    X = np.array(features)
    y = np.array(labels)
    
    print(f"Shape of X before saving: {X.shape}")
    print(f"Shape of y before saving: {y.shape}")
    print(f"Expected feature count: {get_feature_count()}")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch in number of features ({X.shape[0]}) and labels ({y.shape[0]})")
    
    if X.shape[1] != get_feature_count():
        raise ValueError(f"Mismatch in feature count. Expected {get_feature_count()}, got {X.shape[1]}")
    
    np.savez(output_file, X=X, y=y)
    print(f"Dataset saved to {output_file}")
    print(f"Total tracks processed: {len(X)}")

if __name__ == "__main__":
    audio_dir = r"F:\Audio Data Sets\DEAM\MEMD_audio"
    annotations_file = r"F:\Audio Data Sets\DEAM\annotations\annotations.csv"
    output_file = "deam_features.npz"
    num_files = 10  # Set to None to process all files
    prepare_deam_dataset(audio_dir, annotations_file, output_file, num_files)