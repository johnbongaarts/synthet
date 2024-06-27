from src.data_preparation import prepare_dataset

if __name__ == "__main__":
    audio_dir = r"F:\Audio Data Sets\FMA\fma_small"
    metadata_path = r"F:\Audio Data Sets\FMA\fma_metadata\tracks.csv"
    output_path = "fma_features_subset.npz"
    num_files = 1000  # Process up to 1000 files, or set to None to process all available files
    prepare_dataset(audio_dir, metadata_path, output_path, num_files)