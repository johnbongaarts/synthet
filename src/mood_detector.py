import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib
from src.feature_config import get_feature_config

FEATURE_CONFIG = get_feature_config()

class MoodDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, features):
        assert len(features) == FEATURE_CONFIG['TOTAL_FEATURES'], f"Expected {FEATURE_CONFIG['TOTAL_FEATURES']} features, but got {len(features)}"
        X_scaled = self.scaler.transform(features.reshape(1, -1))
        return self.model.predict(X_scaled)[0]

    def save(self, scaler_path, model_path):
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.model, model_path)

    @classmethod
    def load(cls, scaler_path, model_path):
        detector = cls()
        detector.scaler = joblib.load(scaler_path)
        detector.model = joblib.load(model_path)
        return detector

def mood_to_label(valence, arousal):
    if valence > 0.5 and arousal > 0.5:
        return "Happy/Excited"
    elif valence > 0.5 and arousal <= 0.5:
        return "Calm/Relaxed"
    elif valence <= 0.5 and arousal > 0.5:
        return "Angry/Tense"
    else:
        return "Sad/Depressed"