import numpy as np, sys
from PySide6.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QTabWidget, QVBoxLayout, QWidget


from compare_widget import CompareWidget
from control_bar    import ControlBar
from mode_panel     import ModePanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Lux Machina @ MANIBVS")

        self._setupUI()


    def _setupUI(self):
        centralWidget = QWidget()
        layout = QVBoxLayout(centralWidget)

        # --------- Mode panel ---------
        self._modePanel = ModePanel()
        layout.addWidget(self._modePanel, stretch=0)

        # --------- Viewers widget ---------
        self._compareWidget = CompareWidget()
        layout.addWidget(self._compareWidget, stretch=1)

        # ------- Control bar --------
        self._controlBar = ControlBar()
        layout.addWidget(self._controlBar, stretch=0)

        self.setCentralWidget(centralWidget)

    def _onLoadReference(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Load reference image", filter="NumPy files (*.npy)")
        
        if not filePath: return

        try:
            data = np.load(filePath)
            if data.ndim not in [2]:
                raise ValueError("Unsupported image format: expected 2D grayscale or 3D RGB image")

            self._compareWidget.setReference(data)

        except Exception as e:
            print(f"Error loading reference image: {e}")


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()