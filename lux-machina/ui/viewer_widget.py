import numpy as np

from PySide6.QtWidgets import QLabel
from PySide6.QtGui     import QImage, QPixmap
from PySide6.QtCore    import Qt


class ViewerWidget(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)

        # The QImage that will be kept as a reference
        self._qImage = None

    def setImage(self, image: np.ndarray):
        # Normalize the image to the range [0, 255]
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            h, w  = image.shape[:2]

        # Build QImage
        qImage = QImage(image.data, w, h, image.strides[0], QImage.Format_Grayscale8)

        # Keep a reference to the QImage to prevent it from being garbage collected
        self.qImage = qImage

        # Show the image
        pixmap = QPixmap.fromImage(qImage)
        self.setPixmap(
            pixmap.scaled(self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
