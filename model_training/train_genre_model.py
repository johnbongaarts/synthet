import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def train_genre_classifier(data_file, model_output, scaler_output, progress_callback=None):
    if progress_callback:
        progress_callback(0)

    # Load the data
    data = np.load(data_file)
    X, y = data['X'], data['y']

    if progress_callback:
        progress_callback(10)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if progress_callback:
        progress_callback(20)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if progress_callback:
        progress_callback(30)

    # Train the classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    if progress_callback:
        progress_callback(70)

    # Evaluate the classifier
    train_score = clf.score(X_train_scaled, y_train)
    test_score = clf.score(X_test_scaled, y_test)
    print(f"Train accuracy: {train_score:.2f}")
    print(f"Test accuracy: {test_score:.2f}")
    
    if progress_callback:
        progress_callback(80)

    # Save the model and scaler
    joblib.dump(clf, model_output)
    joblib.dump(scaler, scaler_output)
    
    if progress_callback:
        progress_callback(100)

    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    # This block allows you to run the script directly for testing
    data_file = "path/to/your/fma_features_subset.npz"
    model_output = "genre_classifier.joblib"
    scaler_output = "genre_scaler.joblib"
    train_genre_classifier(data_file, model_output, scaler_output)