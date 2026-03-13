import sys
import time
import warp as wp
import numpy as np

from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtGui import QImage, QPixmap


class BufferViewer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def updateFrame(self, buffer):
        # Convert Warp buffer to Numpy array
        if   isinstance(buffer, wp.array): 
            img = buffer.numpy()
        elif isinstance(buffer, np.ndarray):
            img = buffer
        else:
            img = np.asarray(buffer)

        img = (img * 255).astype(np.uint8)
        height, width = img.shape

        qImage = QImage(
            width, height, width,
            QImage.Format_Grayscale8
        )

        self.setPixmap(QPixmap.fromImage(qImage))


@wp.kernel
def noiseKernel(
    frameID: wp.int32,
    buffer: wp.array(dtype=wp.float32, ndim=2)
):
    xID, yID = wp.tid()

    seed = (frameID * buffer.shape[0] + xID) * buffer.shape[1] + yID
    rand = wp.randf(wp.uint32(seed))

    buffer[xID, yID] = rand