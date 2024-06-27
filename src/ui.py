import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QTextEdit, QProgressBar
import os
import pickle
from src import MoodDetector, mood_to_label, analyze_audio, predict_genre

class AudioAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.last_directory = self.load_last_directory()
        self.initUI()

    def load_last_directory(self):
        try:
            with open('last_directory.pkl', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return ''

    def save_last_directory(self, directory):
        with open('last_directory.pkl', 'wb') as f:
            pickle.dump(directory, f)

    def selectFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', self.last_directory, "Wave files (*.wav)")
        if fname:
            self.last_directory = os.path.dirname(fname)
            self.save_last_directory(self.last_directory)
            self.analyzeAudio(fname)

    def initUI(self):
        layout = QVBoxLayout()

        self.btn_select_file = QPushButton('Select WAV File', self)
        self.btn_select_file.clicked.connect(self.selectFile)
        layout.addWidget(self.btn_select_file)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        self.text_results = QTextEdit(self)
        self.text_results.setReadOnly(True)
        layout.addWidget(self.text_results)

        self.setLayout(layout)
        self.setGeometry(300, 300, 400, 300)
        self.setWindowTitle('Audio Analyzer')

    def selectFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', self.last_directory, "Wave files (*.wav)")
        if fname:
            self.last_directory = os.path.dirname(fname)
            self.save_last_directory(self.last_directory)
            self.analyzeAudio(fname)

    def analyzeAudio(self, file_path):
        self.progress_bar.setValue(0)
        features, basic_stats = analyze_audio(file_path, self.progress_bar)
        
        genre_prediction, top_genres = predict_genre(features, 'genre_classifier.joblib', 'genre_scaler.joblib')

        # Mood detection
        mood_detector = MoodDetector.load('mood_scaler.joblib', 'mood_model.joblib')
        valence, arousal = mood_detector.predict(features)
        mood_label = mood_to_label(valence, arousal)

        results = f"File: {file_path}\n\n"
        results += "Basic Audio Statistics:\n"
        for key, value in basic_stats.items():
            results += f"{key}: {value}\n"

        results += f"\nPredicted Genre: {genre_prediction}\n"
        results += "Top 3 Genre Probabilities:\n"
        for genre, prob in top_genres:
            results += f"{genre}: {prob:.2f}\n"

        results += f"\nMood: {mood_label}\n"
        results += f"Valence: {valence:.2f}, Arousal: {arousal:.2f}\n"

        self.text_results.setText(results)
        self.progress_bar.setValue(100)

def run_app():
    app = QApplication(sys.argv)
    ex = AudioAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())