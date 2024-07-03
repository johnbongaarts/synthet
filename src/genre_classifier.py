import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from src.feature_config import get_genre_feature_config

FEATURE_CONFIG = get_genre_feature_config()

def train_genre_classifier(data_file, model_output, scaler_output):
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found. Please run the data preparation script first.")
        return

    data = np.load(data_file)
    X, y = data['X'], data['y']

    if len(X) == 0 or len(y) == 0:
        print("The dataset is empty. Please run the data preparation script to generate a non-empty dataset.")
        return

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate the classifier
    train_score = clf.score(X_train_scaled, y_train)
    test_score = clf.score(X_test_scaled, y_test)
    print(f"Train accuracy: {train_score:.2f}")
    print(f"Test accuracy: {test_score:.2f}")
    
    # Save the model and scaler
    joblib.dump(clf, model_output)
    joblib.dump(scaler, scaler_output)
    print("Model and scaler saved successfully.")

def predict_genre(audio_features, model_file, scaler_file):
    try:
        clf = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
    except FileNotFoundError:
        print("Classifier or scaler not found. Unable to predict genre.")
        return None

    # Use features up to MFCCs for genre classification
    genre_features = audio_features[:FEATURE_CONFIG['TOTAL_FEATURES']]

     # Ensure genre_features is a 2D array
    if len(genre_features.shape) == 1:
        genre_features = genre_features.reshape(1, -1)

    features_scaled = scaler.transform(genre_features)
    genre_prediction = clf.predict(features_scaled)[0]
    genre_probabilities = clf.predict_proba(features_scaled)[0]
    top_genres = sorted(zip(clf.classes_, genre_probabilities), key=lambda x: x[1], reverse=True)[:3]

    return genre_prediction, top_genres