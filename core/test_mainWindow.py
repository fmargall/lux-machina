"""
test_MainWindow.py — Standalone test of the MainWindow class.

Verifies that two BufferDisplay widgets can be composed in a main window,
each receiving independent updates.
"""

import sys
import numpy as np
from pyqtgraph.Qt import QtWidgets, QtCore

from _ui import BufferDisplay, MainWindow


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # ── Build two displays with different configurations ──
    leftDisplay = BufferDisplay(
        title    = "Left (e.g. visualizer)",
        xLabel   = "x (m)",
        yLabel   = "z (m)",
        extent   = (-0.04, 0.04, -0.01, 0.12),
        colormap = "inferno",
    )

    rightDisplay = BufferDisplay(
        title    = "Right (e.g. sensor)",
        xLabel   = "x (m)",
        yLabel   = "y (m)",
        extent   = (-0.1, 0.1, -0.1, 0.1),
        colormap = "viridis",
    )

    # ── Compose them into a main window ──
    window = MainWindow(
        leftDisplay  = leftDisplay,
        rightDisplay = rightDisplay,
        title        = "Test MainWindow",
    )
    window.show()

    # ── Generate independent test patterns for each display ──
    RES_LEFT_W,  RES_LEFT_H  = 512, 832
    RES_RIGHT_W, RES_RIGHT_H = 256, 256

    leftBuffer  = np.zeros((RES_LEFT_W,  RES_LEFT_H),  dtype=np.float32)
    rightBuffer = np.zeros((RES_RIGHT_W, RES_RIGHT_H), dtype=np.float32)

    frame = [0]

    def updateFrame():
        # ── Left: accumulating Gaussian (simulates ray paths) ──
        cx = np.random.uniform(-0.5, 0.5)
        cy = np.random.uniform(-0.5, 0.5)
        sigma = 0.1
        x = np.linspace(-1, 1, RES_LEFT_W)
        y = np.linspace(-1, 1, RES_LEFT_H)
        X, Y = np.meshgrid(x, y, indexing="ij")
        leftBuffer[:] = leftBuffer + np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)).astype(np.float32)

        # ── Right: another accumulating Gaussian, different center ──
        cx = np.random.uniform(-0.3, 0.3)
        cy = np.random.uniform(-0.3, 0.3)
        sigma = 0.05
        x = np.linspace(-1, 1, RES_RIGHT_W)
        y = np.linspace(-1, 1, RES_RIGHT_H)
        X, Y = np.meshgrid(x, y, indexing="ij")
        rightBuffer[:] = rightBuffer + np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)).astype(np.float32)

        # ── Refresh both views together ──
        window.updateViews(
            leftBuffer  = leftBuffer,
            rightBuffer = rightBuffer,
            frameID     = frame[0],
        )
        frame[0] += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(updateFrame)
    timer.start(1)

    sys.exit(app.exec())