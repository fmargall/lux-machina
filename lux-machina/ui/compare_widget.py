import numpy        as np
import cmcrameri.cm as cmc

from PySide6.QtWidgets import QWidget, QHBoxLayout

from .viewer_widget import ViewerWidget


class CompareWidget(QWidget):
    def __init__(self):
        super().__init__()

        self._render    = None
        self._reference = None

        self._setupUI()

    def setRender(self, image: np.ndarray):
        self._render = image
        self._renderView.setImage(image)
        self._update_error()

    def setReference(self, image: np.ndarray):
        self._reference = image
        self._referenceView.setImage(image)
        self._update_error()

    def _setupUI(self):
        layout = QHBoxLayout(self)

        self._renderView     = ViewerWidget()
        self._referenceView  = ViewerWidget()
        self._differenceView = ViewerWidget()

        layout.addWidget(self._renderView)
        layout.addWidget(self._referenceView)
        layout.addWidget(self._differenceView)

    def _update_error(self):

        # Absolute signed error from reference
        error = self._render - self._reference

        maxAbs     = np.max(np.abs(error))
        normalized = 0.5 + 0.5 * error / maxAbs if maxAbs > 0 else np.zeros_like(error)
        normalized = np.clip(normalized, 0.0, 1.0)

        # Build error image with colormap:
        rgb = cmc.vik(normalized)[..., :3]
        rgb = (rgb * 255).astype(np.uint8)
        self._differenceView.setImage(rgb)