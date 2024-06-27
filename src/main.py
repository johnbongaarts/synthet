import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QTextEdit
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QTextEdit, QProgressBar
import librosa
import numpy as np
import os
import pickle
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

class AudioAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.pickle_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'last_directory.pkl')
        print(f"Pickle file path: {self.pickle_path}")
        self.last_directory = self.load_last_directory()
        print(f"Loaded last directory: {self.last_directory}")
        self.train_genre_classifier()
        self.initUI()

    def load_last_directory(self):
        try:
            with open(self.pickle_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"Pickle file not found at {self.pickle_path}, using default")
            return os.path.expanduser('~')  # Use user's home directory as default
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return os.path.expanduser('~')

    def save_last_directory(self, directory):
        try:
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(directory, f)
            print(f"Saved last directory: {directory}")
        except Exception as e:
            print(f"Error saving pickle file: {e}")

    def selectFile(self):
        print(f"Opening file dialog with initial directory: {self.last_directory}")
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', self.last_directory, "Wave files (*.wav)")
        if fname:
            self.last_directory = os.path.dirname(fname)
            print(f"New directory selected: {self.last_directory}")
            self.save_last_directory(self.last_directory)
            print(f"Checking if pickle file was created: {os.path.exists(self.pickle_path)}")
            self.analyzeAudio(fname)
        else:
            print("No file selected")

    def initUI(self):
        layout = QVBoxLayout()

        self.btn_select_file = QPushButton('Select WAV File', self)
        self.btn_select_file.clicked.connect(self.selectFile)
        layout.addWidget(self.btn_select_file)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.text_results = QTextEdit(self)
        self.text_results.setReadOnly(True)
        layout.addWidget(self.text_results)

        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Audio Analyzer')
        self.show()

    def analyzeAudio(self, file_path):
        self.progress_bar.setValue(0)
        y, sr = librosa.load(file_path, duration=120)  # Limit to 2 minutes for quicker analysis
        self.progress_bar.setValue(10)

        # Basic features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0]
        self.progress_bar.setValue(20)

        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
        self.progress_bar.setValue(40)

        # Rhythm features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        self.progress_bar.setValue(60)

        # Timbre features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1)
        mfcc_vars = mfccs.var(axis=1)
        self.progress_bar.setValue(80)

        # Harmonic features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = chroma.mean(axis=1)

        # Calculate style characteristics (example weighting system)
        brightness = (spectral_centroid - 500) / 2000  # Normalized, 0-1 scale
        complexity = np.mean(spectral_contrast[1:]) / 50  # Higher contrast in upper bands
        rhythmic_intensity = np.mean(pulse) / np.max(pulse)
        harmonic_complexity = np.std(chroma_means)
        timbral_richness = np.mean(mfcc_vars[1:])  # Variability in upper MFCCs

        style_vector = {
            "Brightness": brightness,
            "Complexity": complexity,
            "Rhythmic Intensity": rhythmic_intensity,
            "Harmonic Complexity": harmonic_complexity,
            "Timbral Richness": timbral_richness
        }
       
         # Prepare features for genre classification
        features_for_classification = np.concatenate([
            [spectral_centroid, spectral_bandwidth, np.mean(spectral_contrast)],
            mfcc_means,
            chroma_means
        ])

        # Load the classifier and scaler
        try:
            clf = joblib.load('genre_classifier.joblib')
            scaler = joblib.load('genre_scaler.joblib')
        except FileNotFoundError:
            self.train_genre_classifier()
            clf = joblib.load('genre_classifier.joblib')
            scaler = joblib.load('genre_scaler.joblib')

        # Predict genre
        features_scaled = scaler.transform(features_for_classification.reshape(1, -1))
        genre_prediction = clf.predict(features_scaled)[0]
        genre_probabilities = clf.predict_proba(features_scaled)[0]
        top_genres = sorted(zip(clf.classes_, genre_probabilities), key=lambda x: x[1], reverse=True)[:3]

        # Prepare results
        results = f"Tempo: {tempo:.2f} BPM\n"
        results += f"Spectral Centroid: {spectral_centroid:.2f} Hz\n"
        results += f"Spectral Bandwidth: {spectral_bandwidth:.2f} Hz\n"
        results += "\nStyle Characteristics:\n"
        for key, value in style_vector.items():
            results += f"{key}: {value:.2f}\n"
        
        results += f"\nPredicted Genre: {genre_prediction}\n"
        results += "Top 3 Genre Probabilities:\n"
        for genre, prob in top_genres:
            results += f"{genre}: {prob:.2f}\n"

        self.text_results.setText(results)
        self.progress_bar.setValue(100)

    def train_genre_classifier(self):
        data_file = 'fma_features_subset.npz'
    
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
        joblib.dump(clf, 'genre_classifier.joblib')
        joblib.dump(scaler, 'genre_scaler.joblib')   
        print("Model and scaler saved successfully.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AudioAnalyzerApp()
    sys.exit(app.exec_())