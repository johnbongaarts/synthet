from src.data_preparation import prepare_dataset
from src.genre_classifier import train_genre_classifier
from src.ui import run_app

if __name__ == "__main__":
    # Uncomment these lines if you need to prepare the dataset and train the classifier
    # audio_dir = r"F:\Audio Data Sets\FMA\fma_small"
    # metadata_path = r"F:\Audio Data Sets\FMA\fma_metadata\tracks.csv"
    # output_path = "fma_features_subset.npz"
    # num_files = 50  # Process up to 1000 files, or set to None to process all available files
    
    # Prepare dataset
    # prepare_dataset(audio_dir, metadata_path, output_path, num_files)
    
    # Train classifier
    # train_genre_classifier(output_path, 'genre_classifier.joblib', 'genre_scaler.joblib')

    # Run the UI application
    run_app()