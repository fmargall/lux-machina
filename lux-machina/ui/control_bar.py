from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout, QWidget


class ControlBar(QWidget):
    def __init__(self):
        super().__init__()

        self._setupUI()

    def _setupUI(self):
        layout = QVBoxLayout(self)

        # - Separator line -
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)

        self._loadRefButton = QPushButton("Load reference (.npy)")
        self._startButton   = QPushButton("Start")
        self._pauseButton   = QPushButton("Pause")

        hLayout = QHBoxLayout()

        hLayout.addWidget(self._loadRefButton)
        hLayout.addStretch() # Push to the right
        hLayout.addWidget(self._startButton)
        hLayout.addWidget(self._pauseButton)

        layout.addLayout(hLayout)