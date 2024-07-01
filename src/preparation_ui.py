import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QTextEdit, QProgressBar, QLabel, QTabWidget
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
        
        self.fma_audio_dir = QLabel("Audio Directory: Not selected")
        self.fma_metadata = QLabel("Metadata File: Not selected")
        self.fma_output = QLabel("Output File: Not selected")
        
        layout.addWidget(self.fma_audio_dir)
        layout.addWidget(self.fma_metadata)
        layout.addWidget(self.fma_output)
        
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
        
        self.genre_data_file = QLabel("Data File: Not selected")
        self.genre_model_output = QLabel("Model Output: Not selected")
        self.genre_scaler_output = QLabel("Scaler Output: Not selected")
        
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
        
        self.deam_audio_dir = QLabel("Audio Directory: Not selected")
        self.deam_annotations = QLabel("Annotations File: Not selected")
        self.deam_output = QLabel("Output File: Not selected")
        
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
        
        self.mood_data_file = QLabel("Data File: Not selected")
        self.mood_model_output = QLabel("Model Output: Not selected")
        self.mood_scaler_output = QLabel("Scaler Output: Not selected")
        
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
        
        try:
            prepare_fma_dataset(audio_dir, metadata_path, output_path, progress_callback=self.updateProgress)
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