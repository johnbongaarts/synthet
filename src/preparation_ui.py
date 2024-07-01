import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QProgressBar, QLabel, QTabWidget, QLineEdit
from PyQt5.QtCore import Qt
import os
from model_training.prepare_fma import prepare_fma_dataset
from model_training.train_genre_model import train_genre_classifier
from model_training.prepare_deam import prepare_deam_dataset
from model_training.train_mood_model import train_mood_model

class PreparationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        
        # Create tabs
        tabs = QTabWidget()
        tabs.addTab(self.createFMATab(), "FMA Dataset")
        tabs.addTab(self.createGenreModelTab(), "Genre Model")
        tabs.addTab(self.createDEAMTab(), "DEAM Dataset")
        tabs.addTab(self.createMoodModelTab(), "Mood Model")
        
        layout.addWidget(tabs)
        
        self.setLayout(layout)
        self.setWindowTitle('Data Preparation and Model Training')
        self.setGeometry(100, 100, 600, 400)

    def createFMATab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        default_fma_audio = os.path.expanduser("F:/Audio Data Sets/FMA/fma_small")
        default_fma_metadata = os.path.expanduser("F:/Audio Data Sets/FMA/fma_metadata/tracks.csv")
        default_fma_output = os.path.expanduser("F:/Audio Data Sets/FMA/fma_small_features.npz")

        self.fma_audio_dir = QLabel(f"Audio Directory: {default_fma_audio}")
        self.fma_metadata = QLabel(f"Metadata File: {default_fma_metadata}")
        self.fma_output = QLabel(f"Output File: {default_fma_output}")

        layout.addWidget(self.fma_audio_dir)
        layout.addWidget(self.fma_metadata)
        layout.addWidget(self.fma_output)

        # Add input for number of files
        num_files_layout = QHBoxLayout()
        num_files_label = QLabel("Number of files to process (blank for all):")
        self.num_files_input = QLineEdit()
        self.num_files_input.setPlaceholderText("Leave blank to process all files")
        num_files_layout.addWidget(num_files_label)
        num_files_layout.addWidget(self.num_files_input)
        layout.addLayout(num_files_layout)
        
        btn_audio = QPushButton("Select Audio Directory")
        btn_audio.clicked.connect(lambda: self.selectDirectory(self.fma_audio_dir, "Audio Directory"))
        
        btn_metadata = QPushButton("Select Metadata File")
        btn_metadata.clicked.connect(lambda: self.selectFile(self.fma_metadata, "Metadata File", "CSV files (*.csv)"))
        
        btn_output = QPushButton("Select Output File")
        btn_output.clicked.connect(lambda: self.saveFile(self.fma_output, "Output File", "NPZ files (*.npz)"))
        
        btn_prepare = QPushButton("Prepare FMA Dataset")
        btn_prepare.clicked.connect(self.prepareFMA)
        
        layout.addWidget(btn_audio)
        layout.addWidget(btn_metadata)
        layout.addWidget(btn_output)
        layout.addWidget(btn_prepare)
        
        self.fma_progress = QProgressBar()
        layout.addWidget(self.fma_progress)
        
        widget.setLayout(layout)
        return widget

    def createGenreModelTab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        default_genre_data = os.path.expanduser("./datasets/fma_features.npz")
        default_genre_model = os.path.expanduser("./models/genre_classifier.joblib")
        default_genre_scaler = os.path.expanduser("./models/genre_scaler.joblib")

        self.genre_data_file = QLabel(f"Data File: {default_genre_data}")
        self.genre_model_output = QLabel(f"Model Output: {default_genre_model}")
        self.genre_scaler_output = QLabel(f"Scaler Output: {default_genre_scaler}")
        
        layout.addWidget(self.genre_data_file)
        layout.addWidget(self.genre_model_output)
        layout.addWidget(self.genre_scaler_output)
        
        btn_data = QPushButton("Select Data File")
        btn_data.clicked.connect(lambda: self.selectFile(self.genre_data_file, "Data File", "NPZ files (*.npz)"))
        
        btn_model = QPushButton("Select Model Output")
        btn_model.clicked.connect(lambda: self.saveFile(self.genre_model_output, "Model Output", "Joblib files (*.joblib)"))
        
        btn_scaler = QPushButton("Select Scaler Output")
        btn_scaler.clicked.connect(lambda: self.saveFile(self.genre_scaler_output, "Scaler Output", "Joblib files (*.joblib)"))
        
        btn_train = QPushButton("Train Genre Model")
        btn_train.clicked.connect(self.trainGenreModel)
        
        layout.addWidget(btn_data)
        layout.addWidget(btn_model)
        layout.addWidget(btn_scaler)
        layout.addWidget(btn_train)
        
        self.genre_progress = QProgressBar()
        layout.addWidget(self.genre_progress)
        
        widget.setLayout(layout)
        return widget

    def createDEAMTab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        default_deam_audio = os.path.expanduser("F:/Audio Data Sets/DEAM/MEMD_audio")
        default_deam_annotations = os.path.expanduser("F:/DEAM/annotations/annotations.csv")
        default_deam_output = os.path.expanduser("./datasets/deam_features.npz")

        self.deam_audio_dir = QLabel(f"Audio Directory: {default_deam_audio}")
        self.deam_annotations = QLabel(f"Annotations File: {default_deam_annotations}")
        self.deam_output = QLabel(f"Output File: {default_deam_output}")
        
        layout.addWidget(self.deam_audio_dir)
        layout.addWidget(self.deam_annotations)
        layout.addWidget(self.deam_output)
        
        btn_audio = QPushButton("Select Audio Directory")
        btn_audio.clicked.connect(lambda: self.selectDirectory(self.deam_audio_dir, "Audio Directory"))
        
        btn_annotations = QPushButton("Select Annotations File")
        btn_annotations.clicked.connect(lambda: self.selectFile(self.deam_annotations, "Annotations File", "CSV files (*.csv)"))
        
        btn_output = QPushButton("Select Output File")
        btn_output.clicked.connect(lambda: self.saveFile(self.deam_output, "Output File", "NPZ files (*.npz)"))
        
        btn_prepare = QPushButton("Prepare DEAM Dataset")
        btn_prepare.clicked.connect(self.prepareDEAM)
        
        layout.addWidget(btn_audio)
        layout.addWidget(btn_annotations)
        layout.addWidget(btn_output)
        layout.addWidget(btn_prepare)
        
        self.deam_progress = QProgressBar()
        layout.addWidget(self.deam_progress)
        
        widget.setLayout(layout)
        return widget

    def createMoodModelTab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        default_mood_data = os.path.expanduser("./datasets/deam_features.npz")
        default_mood_model = os.path.expanduser("./models/mood_model.joblib")
        default_mood_scaler = os.path.expanduser("./models/mood_scaler.joblib")

        self.mood_data_file = QLabel(f"Data File: {default_mood_data}")
        self.mood_model_output = QLabel(f"Model Output: {default_mood_model}")
        self.mood_scaler_output = QLabel(f"Scaler Output: {default_mood_scaler}")
        
        layout.addWidget(self.mood_data_file)
        layout.addWidget(self.mood_model_output)
        layout.addWidget(self.mood_scaler_output)
        
        btn_data = QPushButton("Select Data File")
        btn_data.clicked.connect(lambda: self.selectFile(self.mood_data_file, "Data File", "NPZ files (*.npz)"))
        
        btn_model = QPushButton("Select Model Output")
        btn_model.clicked.connect(lambda: self.saveFile(self.mood_model_output, "Model Output", "Joblib files (*.joblib)"))
        
        btn_scaler = QPushButton("Select Scaler Output")
        btn_scaler.clicked.connect(lambda: self.saveFile(self.mood_scaler_output, "Scaler Output", "Joblib files (*.joblib)"))
        
        btn_train = QPushButton("Train Mood Model")
        btn_train.clicked.connect(self.trainMoodModel)
        
        layout.addWidget(btn_data)
        layout.addWidget(btn_model)
        layout.addWidget(btn_scaler)
        layout.addWidget(btn_train)
        
        self.mood_progress = QProgressBar()
        layout.addWidget(self.mood_progress)
        
        widget.setLayout(layout)
        return widget

    def selectDirectory(self, label, title):
        directory = QFileDialog.getExistingDirectory(self, f"Select {title}")
        if directory:
            label.setText(f"{title}: {directory}")

    def selectFile(self, label, title, file_filter):
        file, _ = QFileDialog.getOpenFileName(self, f"Select {title}", "", file_filter)
        if file:
            label.setText(f"{title}: {file}")

    def saveFile(self, label, title, file_filter):
        file, _ = QFileDialog.getSaveFileName(self, f"Select {title}", "", file_filter)
        if file:
            label.setText(f"{title}: {file}")

    def prepareFMA(self):
        audio_dir = self.fma_audio_dir.text().split(": ")[1]
        metadata_path = self.fma_metadata.text().split(": ")[1]
        output_path = self.fma_output.text().split(": ")[1]
        
        # Get the number of files to process
        num_files_text = self.num_files_input.text().strip()
        num_files = int(num_files_text) if num_files_text else None
        
        try:
            prepare_fma_dataset(audio_dir, metadata_path, output_path, num_files=num_files, progress_callback=self.updateProgress)
            self.showMessage("FMA dataset preparation completed successfully!")
        except Exception as e:
            self.showMessage(f"Error preparing FMA dataset: {str(e)}")

    def trainGenreModel(self):
        data_file = self.genre_data_file.text().split(": ")[1]
        model_output = self.genre_model_output.text().split(": ")[1]
        scaler_output = self.genre_scaler_output.text().split(": ")[1]
        
        try:
            train_genre_classifier(data_file, model_output, scaler_output, progress_callback=self.updateProgress)
            self.showMessage("Genre model training completed successfully!")
        except Exception as e:
            self.showMessage(f"Error training genre model: {str(e)}")

    def prepareDEAM(self):
        audio_dir = self.deam_audio_dir.text().split(": ")[1]
        annotations_file = self.deam_annotations.text().split(": ")[1]
        output_file = self.deam_output.text().split(": ")[1]
        
        try:
            prepare_deam_dataset(audio_dir, annotations_file, output_file, progress_callback=self.updateProgress)
            self.showMessage("DEAM dataset preparation completed successfully!")
        except Exception as e:
            self.showMessage(f"Error preparing DEAM dataset: {str(e)}")

    def trainMoodModel(self):
        data_file = self.mood_data_file.text().split(": ")[1]
        model_output = self.mood_model_output.text().split(": ")[1]
        scaler_output = self.mood_scaler_output.text().split(": ")[1]
        
        try:
            train_mood_model(data_file, model_output, scaler_output, progress_callback=self.updateProgress)
            self.showMessage("Mood model training completed successfully!")
        except Exception as e:
            self.showMessage(f"Error training mood model: {str(e)}")

    def updateProgress(self, value):
        self.fma_progress.setValue(value)
        self.genre_progress.setValue(value)
        self.deam_progress.setValue(value)
        self.mood_progress.setValue(value)

    def showMessage(self, message):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Information", message)