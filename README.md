1. Install required dependencies using `pip install -r requirements.txt`.

2. Prepare your dataset by running `python src/data_preparation.py` with appropriate audio directory and metadata file paths.

3. Train the genre classifier by executing `python src/genre_classifier.py`.

4. Train the mood detector by running `python src/mood_detector.py`.

5. Launch the main application by running `python main.py`.

6. Use the application by clicking "Select WAV File" and choosing an audio file for analysis.

7. View the analysis results, including genre prediction and mood detection, in the application window.

8. Retrain the genre classifier if you modify the feature extraction process or add new genres to your dataset.

9. Retrain the mood detector if you change the feature set or want to improve mood prediction accuracy.

10. To update the feature set, modify `src/feature_config.py` and retrain both models.