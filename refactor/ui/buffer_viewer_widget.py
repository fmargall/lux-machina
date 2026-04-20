import numpy as np

from PySide6.QtWidgets import QLabel
from PySide6.QtGui     import QImage, QPixmap
from PySide6.QtCore    import Qt


class BufferViewerWidget(QLabel):
    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)

        # The QImage that will be kept as a reference
        self._qImage = None

    def setImage(self, image):
        # Image can be either directly in bytes or a NumPy array
        if type(image) == bytes:
            qImage = QImage.fromData(image)

        else:
            # Normalize the image to the range [0, 255]
            if image.dtype != np.uint8:
                image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

            h, w  = image.shape[:2]

            # Build QImage
            if image.ndim == 2: # Gray scale
                qImage = QImage(image.data, w, h, image.strides[0], QImage.Format_Grayscale8)
            elif image.ndim == 3 and image.shape[2] == 3: # RGB
                qImage = QImage(image.data, w, h, image.strides[0], QImage.Format_RGB888)
            else:
                raise ValueError("Unsupported image format: expected 2D grayscale or 3D RGB image")

        # Keep a reference to the QImage to prevent it from being garbage collected
        self._qImage = qImage

        # Show the image
        pixmap = QPixmap.fromImage(qImage)
        self.setPixmap(
            pixmap.scaled(self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
