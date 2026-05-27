"""
test_BufferDisplay.py — Standalone test of the BufferDisplay widget.

Verifies that the widget displays a buffer, refreshes it, and applies transforms
correctly.
"""

import sys
import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore

from _ui import BufferDisplay

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Create a widget with log1p transform (typical for ray density visualization)
    widget = BufferDisplay(
        title     = "Test pattern",
        xLabel    = "x",
        yLabel    = "y",
        extent    = (-1.0, 1.0, -1.0, 1.0),
        colormap  = "inferno",
        transform = "log1p",
    )
    widget.resize(800, 800)
    widget.show()

    # ── Generate an evolving test pattern: an accumulating Gaussian ──
    RES = 256
    buffer = np.zeros((RES, RES), dtype=np.float32)

    frame = [0]   # mutable container for the closure

    def updateFrame():
        # Add a random Gaussian splat each call to simulate accumulation
        cx = np.random.uniform(-0.5, 0.5)
        cy = np.random.uniform(-0.5, 0.5)
        sigma = 0.1

        x = np.linspace(-1, 1, RES)
        y = np.linspace(-1, 1, RES)
        X, Y = np.meshgrid(x, y, indexing="ij")
        gaussian = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

        # Accumulate
        buffer[:] = buffer + gaussian.astype(np.float32)

        # Refresh the widget
        widget.update(buffer)

        frame[0] += 1
        widget.setWindowTitle(f"Test pattern — frame {frame[0]}")

    # Update every 100 ms
    timer = QtCore.QTimer()
    timer.timeout.connect(updateFrame)
    timer.start(1000)

    sys.exit(app.exec())