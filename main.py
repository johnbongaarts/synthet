
from src import prepare_dataset, train_genre_classifier, MoodDetector, run_app, TOTAL_FEATURES
import numpy as np

def train_mood_detector(data_path, scaler_path, model_path):
    data = np.load(data_path)
    X = data['X']
    
    print(f"Shape of X: {X.shape}")
    print(f"TOTAL_FEATURES: {TOTAL_FEATURES}")
    
    assert X.shape[1] == TOTAL_FEATURES, "Unexpected number of features in dataset"
    
    # For this example, we'll generate random mood labels
    # In a real scenario, you'd use actual mood labels
    y = np.random.rand(X.shape[0], 2)  # Random valence and arousal values
    
    detector = MoodDetector()
    detector.train(X, y)
    detector.save(scaler_path, model_path)
    print("Mood detector trained and saved.")

def inspect_dataset(data_path):
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"Number of features: {X.shape[1]}")
    
    # Print the first few rows of X to see the feature values
    print("\nFirst few rows of X:")
    print(X[:5])
    
    # Print unique values in y to see what labels we have
    print("\nUnique values in y:")
    print(np.unique(y))

    return X.shape[1]  # Return the number of features

if __name__ == "__main__":
    data_path = 'fma_features_subset.npz'
    num_features = inspect_dataset(data_path)
    
    print(f"\nTotal number of features in the dataset: {num_features}")

    # Uncomment these lines if you need to prepare the dataset and train the classifier
    # audio_dir = r"F:\Audio Data Sets\FMA\fma_small"
    # metadata_path = r"F:\Audio Data Sets\FMA\fma_metadata\tracks.csv"
    # output_path = "fma_features_subset.npz"
    # num_files = 50  # Process up to 1000 files, or set to None to process all available files
   
    # Prepare dataset
    # prepare_dataset(audio_dir, metadata_path, output_path, num_files)
    
    # Train classifier
    # train_genre_classifier(output_path, 'genre_classifier.joblib', 'genre_scaler.joblib')

    # Train mood detector
    train_mood_detector('fma_features_subset.npz', 'mood_scaler.joblib', 'mood_model.joblib')

    # Run the UI application
    run_app()