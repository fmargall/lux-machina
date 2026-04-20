from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QVBoxLayout
from PySide6.QtCore    import Signal

from buffer_viewer_widget  import BufferViewerWidget
from camera_control_widget import CameraControlWidget

class CameraCommandsWidget(QWidget):
    startStopLiveViewClicked = Signal()
    shotClicked              = Signal()

    def __init__(self):
        super().__init__()

        self._isLive = False

        self.liveButton = QPushButton("Start Live View")
        self.shotButton = QPushButton("Shot")

        layout = QHBoxLayout()
        layout.addWidget(self.liveButton)
        layout.addWidget(self.shotButton)
        self.setLayout(layout)

        self.liveButton.clicked.connect(self._toggleLive)
        self.shotButton.clicked.connect(self.shotClicked)

    def _toggleLive(self):
        self._isLive = not self._isLive

        if self._isLive:
            self.liveButton.setText("Stop Live View")
        else:
            self.liveButton.setText("Start Live View")

        self.startStopLiveViewClicked.emit()

    def setLiveState(self, isLive: bool):
        """Allow to synchronize camera state from the outside"""
        self._isLive = isLive
        self.liveButton.setText(
            "Stop Live View" if isLive else "Start Live View"
        )


class CameraWidget(QWidget):
    frameReady = Signal(object)

    def __init__(self, camera):
        super().__init__()

        self.camera = camera
        self._stream = None
        self._isLive = False

        # Widgets
        self.controls  = CameraControlWidget(camera)
        self.commands  = CameraCommandsWidget()
        self.viewer    = BufferViewerWidget()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.commands)
        layout.addWidget(self.controls)
        layout.addWidget(self.viewer)
        layout.setStretch(2, 1)
        self.setLayout(layout)

        # From UI to camera
        self.controls.exposureChanged.connect(
            lambda v: setattr(self.camera, "shutterSpeed", v)
        )
        self.controls.isoChanged.connect(
            lambda v: setattr(self.camera, "isoSpeed", v)
        )
        self.controls.apertureChanged.connect(
            lambda v: setattr(self.camera, "aperture", v)
        )

        # Button
        self.commands.startStopLiveViewClicked.connect(self._toggleLiveView)
        self.commands.shotClicked.connect(self._shot)

        # Thread-safe stream to viewer
        self.frameReady.connect(self.viewer.setImage)

    def _toggleLiveView(self):
        if not self._isLive:
            self._startLiveView()
        else:
            self._stopLiveView()

        self._isLive = not self._isLive
        self.commands.setLiveState(self._isLive)


    def _startLiveView(self):
        self._stream = self.camera.liveViewStream(callback=self._onFrame)


    def _stopLiveView(self):
        if self._stream:
            self._stream.stop()
            self._stream = None

    def _shot(self):
        # ⚠️ blocking → idéalement à mettre dans un thread plus tard
        self.camera.shot()

    def _onFrame(self, frame):
        self.frameReady.emit(frame)