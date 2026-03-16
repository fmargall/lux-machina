import numpy as np
import warp  as wp

from PySide6.QtWidgets import QLabel
from PySide6.QtGui     import QImage, QPixmap

import cmcrameri.cm as cmc

class BufferImage(QLabel):
    def __init__(self, parent=None, colormap=cmc.grayC):
        super().__init__(parent)
        self.colormap = colormap

    def updateFrame(self, buffer):
        # Convert Warp buffer to Numpy array
        if   isinstance(buffer, wp.array): 
            img = buffer.numpy()
        elif isinstance(buffer, np.ndarray):
            img = buffer
        else:
            img = np.asarray(buffer)

        if self.colormap == cmc.grayC:
            img = (img * 255).astype(np.uint8)
            height, width = img.shape
            bytesPerLine = width

            qImage = QImage(
                width, height, bytesPerLine,
                QImage.Format_Grayscale8
            )

        else:
            # Remove alpha and apply colormap
            rgb = self.colormap(img)[..., :3]
            rgb = (rgb * 255).astype(np.uint8)

            height, width, _ = rgb.shape
            bytesPerLine = 3 * width

            qImage = QImage(
                width, height, bytesPerLine,
                QImage.Format_RGB888
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