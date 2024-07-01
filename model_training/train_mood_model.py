import numpy as np
from src.mood_detector import MoodDetector

def train_mood_model(data_file, model_output, scaler_output):
    data = np.load(data_file)
    X, y = data['X'], data['y']

    detector = MoodDetector()
    detector.train(X, y)
    detector.save(scaler_output, model_output)
    print(f"Mood detection model trained and saved to {model_output} and {scaler_output}")

if __name__ == "__main__":
    data_file = "deam_features.npz"
    model_output = "../models/mood_model.joblib"
    scaler_output = "../models/mood_scaler.joblib"
    train_mood_model(data_file, model_output, scaler_output)