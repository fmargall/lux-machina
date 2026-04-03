from PySide6.QtCore import QObject, QTimer
from PySide6.QtCore import Slot


class RenderWorker(QObject):
    def __init__(self, engine):
        super().__init__()

        self._engine =  engine
        self._mode   = "render"

        self._timer = None

    def setMode(self, mode: str):
        self._mode = mode

    @Slot()
    def start(self):
        if self._timer is None:
            self._timer = QTimer()

        # Run maximum speed
        self._timer.start(0)

    @Slot()
    def stop(self):
        if self._timer is not None:
            self._timer.stop()


    def _step(self):
        if self._mode == "render":
            image = self._engine.render_frame()

        elif self._mode == "optim":
            image = self._engine.optimize_step()
