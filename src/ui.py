import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QTextEdit, QProgressBar, QLabel, QSplitter, QDesktopWidget
import os
import pickle
from src import MoodDetector, mood_to_label, analyze_audio, predict_genre
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import librosa
import numpy as np
from PyQt5.QtCore import Qt

class WaveformWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure, self.ax = plt.subplots(facecolor='#2D2D2D')
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # self.setMinimumSize(500, 300)  # Width, Height

        self.setup_plot()

    def setup_plot(self):
        self.ax.clear()
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')

        # Change the color of the box around the plot
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')

        self.ax.set_xlabel('Time (Seconds)', color='white')
        self.ax.set_ylabel('Amplitude (Peak RMS)', color='white')
        self.ax.set_title('Audio Waveform', color='white')
        self.figure.patch.set_facecolor('#2D2D2D')
        self.ax.set_facecolor('#2D2D2D')
        self.ax.set_aspect('auto')
        self.canvas.draw()

    def plot_waveform(self, audio_path):
        y, sr = librosa.load(audio_path)
        time = np.linspace(0, len(y) / sr, num=len(y))

        # Downsample for visualization
        downsample_factor = max(1, len(y) // 1000)
        y_downsampled = y[::downsample_factor]
        time_downsampled = time[::downsample_factor]
        
        self.ax.clear()
        self.ax.plot(time_downsampled, y_downsampled, color='#4CAF50')
        self.ax.set_facecolor('#2D2D2D')
        self.ax.tick_params(axis='x', color='white')
        self.ax.tick_params(axis='y', color='white')
        self.ax.set_xlabel('Time (Seconds)', color='white')
        self.ax.set_ylabel('Amplitude (Peak RMS)', color='white')
        self.ax.set_title('Audio Waveform', color='white')
        self.ax.set_aspect('auto')

        # Change the color of the box around the plot
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')

        self.figure.tight_layout()
        self.canvas.draw()


class MoodPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        # self.setMinimumSize(500, 300)  # Width, Height
        self.setup_plot()

    def setup_plot(self):
        self.ax.clear()
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        
        # Change the color of the box around the plot
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')

        self.ax.set_xlabel('Valence', color='white')
        self.ax.set_ylabel('Arousal', color='white')
        self.ax.set_title('Mood Characteristics', color='white')
        self.figure.patch.set_facecolor('#2D2D2D')
        self.ax.set_facecolor('#2D2D2D')
        self.canvas.draw()

    def plot_mood(self, valence, arousal):
        self.ax.clear()
        self.ax.set_facecolor('#2D2D2D')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')

        # Calculate plot limits
        max_abs_val = max(abs(valence), abs(arousal), 1)  # Ensure at least -1 to 1 range
        limit = max_abs_val * 1.1  # Add 10% padding

        # Set aspect ratio to 'auto' to maintain rectangular shape
        self.ax.set_aspect('auto')

        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)

        # Change the color of the box around the plot
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')

        # Create gradient backgrounds
        n_bins = 100
        x = np.linspace(-limit, limit, n_bins)
        y = np.linspace(-limit, limit, n_bins)
        X, Y = np.meshgrid(x, y)

        # Define colors for each quadrant (corrected order)
        colors = ['#FF6666', '#66FF66', '#FFFF66', '#6666FF']  # Red, Green, Yellow, Blue
        alphas = [0.2, 0.2, 0.2, 0.2]  # Adjust these values to change the intensity of the gradients

        for i, (color, alpha) in enumerate(zip(colors, alphas)):
            mask = np.ones((n_bins, n_bins), dtype=bool)
            if i in [0, 1]:  # Top quadrants
                mask[Y < 0] = False
            else:  # Bottom quadrants
                mask[Y >= 0] = False
            if i in [0, 3]:  # Left quadrants
                mask[X >= 0] = False
            else:  # Right quadrants
                mask[X < 0] = False

            Z = np.sqrt(X**2 + Y**2) * mask
            cmap = mcolors.LinearSegmentedColormap.from_list("", ['#2D2D2D', color])
            self.ax.imshow(Z, extent=[-limit, limit, -limit, limit], origin='lower', cmap=cmap, alpha=alpha, aspect='auto')
        
        # Plot axes
        self.ax.axhline(y=0, color='#808080', linestyle='--', alpha=0.5)
        self.ax.axvline(x=0, color='#808080', linestyle='--', alpha=0.5)

        # Add labels for quadrants
        self.ax.text(limit*0.5, limit*0.5, 'Happy/Excited', ha='center', va='center', color='white')
        self.ax.text(-limit*0.5, limit*0.5, 'Angry/Tense', ha='center', va='center', color='white')
        self.ax.text(-limit*0.5, -limit*0.5, 'Sad/Depressed', ha='center', va='center', color='white')
        self.ax.text(limit*0.5, -limit*0.5, 'Calm/Relaxed', ha='center', va='center', color='white')

        # Plot the mood point
        self.ax.plot(valence, arousal, marker='o', color='white', markersize=10, markeredgecolor='black')

        # Add value annotations
        self.ax.annotate(f'({valence:.2f}, {arousal:.2f})', (valence, arousal), 
                        xytext=(5, 5), textcoords='offset points',
                        color='white', 
                        bbox=dict(boxstyle='round,pad=0.5', fc='#363636', ec='white', alpha=0.7))

        self.ax.set_xlabel('Valence', color='white')
        self.ax.set_ylabel('Arousal', color='white')
        self.ax.set_title('Mood Characteristics', color='white')

        self.figure.tight_layout()
        self.canvas.draw()

class AudioAnalyzerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.last_directory = self.load_last_directory()
        self.initUI()
        self.apply_dark_mode_style()
        #self.apply_styles()
        #self.setMinimumSize(400, 300)  # Width, Height

    def position_window(self):
        # Get the screen geometry
        screen = QDesktopWidget().screenNumber(QDesktopWidget().cursor().pos())
        screen_geometry = QDesktopWidget().availableGeometry(screen)

        # Adjust window size if it's too large for the screen
        window_width = min(self.width(), screen_geometry.width() * 0.9)  # 90% of screen width
        window_height = min(self.height(), screen_geometry.height() * 0.9)  # 90% of screen height
        self.resize(window_width, window_height)

        # Calculate the top right position
        left_x = 0 #This will place it to the left
        top_y = 0  # This will place it at the top

        # Move the window
        self.move(left_x, top_y)

    # This is for the file browser memory feature, but should probably go somewhere else
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
            self.file_name_label.setText(os.path.basename(fname))
            self.analyzeAudio(fname)

    def apply_dark_mode_style(self):
        dark_style = """
        QWidget {
            background-color: #2D2D2D;
            color: #FFFFFF;
            font-family: Arial;
        }
        QPushButton {
            background-color: #4CAF50;
            border: none;
            color: white;
            padding: 5px 15px;
            text-align: center;
            font-size: 14px;
            margin: 4px 2px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QTextEdit, QLabel {
            background-color: #363636;
            border: 1px solid #4CAF50;
            border-radius: 4px;
            padding: 5px;
        }
        QProgressBar {
            border: 2px solid #4CAF50;
            border-radius: 5px;
            text-align: center;
            background-color: #363636
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
        }
        QLabel, QTextEdit {
        color: #FFFFFF;
        }
        """
        self.setStyleSheet(dark_style)

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-size: 14px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 10px 20px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 5px;
            }
            QLabel {
                font-weight: bold;
            }
        """)
        self.text_results.setStyleSheet("""
        QTextEdit {
            font-size: 16px;
            line-height: 1.5;
        }
    """)

    def initUI(self):
        main_layout = QHBoxLayout()

        # Create a splitter for flexible resizing
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #4CAF50;
            }
            QSplitter::handle:horizontal {
                width: 2px;
            }
            QSplitter::handle:vertical {
                height: 2px;
            }
        """)

        # Left side (plots)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # File selection area
        file_layout = QHBoxLayout()
        self.btn_select_file = QPushButton('Select WAV File', self)
        self.btn_select_file.clicked.connect(self.selectFile)
        file_layout.addWidget(self.btn_select_file)
        self.progress_bar = QProgressBar(self)
        file_layout.addWidget(self.progress_bar)
        left_layout.addLayout(file_layout)

        # File name display
        self.file_name_label = QLabel("No file selected", self)
        self.file_name_label.setAlignment(Qt.AlignCenter)
        self.file_name_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(self.file_name_label)

        # Plots
        self.waveform_widget = WaveformWidget(self)
        left_layout.addWidget(self.waveform_widget)
        self.mood_plot_widget = MoodPlotWidget(self)
        left_layout.addWidget(self.mood_plot_widget)

        splitter.addWidget(left_widget)

        # Right side (results)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.addWidget(QLabel("Analysis Results:"))
        self.text_results = QTextEdit(self)
        self.text_results.setReadOnly(True)
        right_layout.addWidget(self.text_results)

        splitter.addWidget(right_widget)

        # Set initial sizes for splitter
        splitter.setSizes([2 * self.width() // 3, self.width() // 3])

        main_layout.addWidget(splitter)

        self.setLayout(main_layout)
        self.setGeometry(100, 100, 1400, 1000)  # Increased width
        self.setWindowTitle('Synthet Audio Intelligence')
        self.position_window()

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

        self.waveform_widget.plot_waveform(file_path)
        self.mood_plot_widget.plot_mood(valence, arousal)
        self.text_results.setText(results)
        self.progress_bar.setValue(100)

def run_app():
    app = QApplication(sys.argv)
    ex = AudioAnalyzerApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_app()