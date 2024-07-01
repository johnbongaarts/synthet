import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

from src.ui import AudioAnalyzerApp
from src.preparation_ui import PreparationApp

class ChoiceWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        btn_main = QPushButton('Run Audio Analyzer', self)
        btn_main.clicked.connect(self.run_main_app)
        layout.addWidget(btn_main)

        btn_prep = QPushButton('Run Data Preparation', self)
        btn_prep.clicked.connect(self.run_prep_app)
        layout.addWidget(btn_prep)

        self.setLayout(layout)
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Choose Application')


    def run_main_app(self):
        self.hide()
        self.main_app = AudioAnalyzerApp()
        self.main_app.show()

    def run_prep_app(self):
        self.hide()
        self.prep_app = PreparationApp()
        self.prep_app.show()

def main():
    app = QApplication(sys.argv)
    ex = ChoiceWindow()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()