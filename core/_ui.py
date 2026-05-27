"""
_ui.py — Reusable building blocks for the simulation UI.

This module provides the base widget for displaying a 2D buffer in real time.
Higher-level components (full main windows, control panels) are built by
composing this widget.
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore


class BufferDisplay(QtWidgets.QWidget):
    """
    A self-contained widget displaying a 2D numpy buffer with real-time refresh.

    Use it for any 2D array you want to visualize: a sensor image, a debug
    visualization, a heatmap, etc. The widget knows nothing about the buffer's
    semantics — it just displays whatever you pass to .update().

    Args:
        title    : Window/plot title.
        xLabel   : Label of the horizontal axis.
        yLabel   : Label of the vertical axis.
        extent   : (xMin, xMax, yMin, yMax) — physical coordinates of the buffer.
                   If None, pixel coordinates are used.
        colormap : Name of a pyqtgraph colormap ("inferno", "viridis", "grays", ...).
        transform: Pre-display transformation applied to the buffer.
                   "linear" : no transformation.
                   "log1p"  : log(1 + x), useful for high dynamic range.
                   "sqrt"   : sqrt(x), gentle compression.

    Example:
        widget = BufferDisplay(
            title    = "Ray paths (xz plane)",
            xLabel   = "x (m)",
            yLabel   = "z (m)",
            extent   = (-0.04, 0.04, -0.01, 0.12),
            colormap = "inferno",
            transform= "log1p",
        )
        widget.show()

        for frame in range(N_FRAMES):
            # ... compute bufferArray ...
            widget.update(bufferArray)
    """

    # Set of valid transformations
    _VALID_TRANSFORMS = ("linear", "log1p", "sqrt")


    def __init__(
        self,
        title    : str        = "",
        xLabel   : str        = "",
        yLabel   : str        = "",
        extent   : tuple|None = None,
        colormap : str        = "viridis",
        transform: str        = "linear",
        parent              = None,
    ):
        super().__init__(parent)

        # ── Validate the transform argument ──
        if transform not in self._VALID_TRANSFORMS:
            raise ValueError(
                f"Unknown transform '{transform}'. "
                f"Valid options: {self._VALID_TRANSFORMS}."
            )
        self._transformName = transform

        # ── Layout: a single GraphicsLayoutWidget containing one plot ──
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._graphicsWidget = pg.GraphicsLayoutWidget()
        layout.addWidget(self._graphicsWidget)

        # ── Configure the plot ──
        self._plot = self._graphicsWidget.addPlot()
        self._plot.setAspectLocked(True)
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.setMenuEnabled(False)
        # Disable padding around the data
        self._plot.getViewBox().setDefaultPadding(0.0)

        # ── Create the image item ──
        self._imageItem = pg.ImageItem()
        self._plot.addItem(self._imageItem)

        # Apply colormap
        self._imageItem.setColorMap(pg.colormap.get(colormap))

        # ── Set the image extent if provided ──
        if extent is not None:
            xMin, xMax, yMin, yMax = extent
            self._imageItem.setRect(QtCore.QRectF(
                xMin,
                yMin,
                xMax - xMin,
                yMax - yMin,
            ))


    def update(self, bufferArray: np.ndarray):
        """
        Refresh the display with new buffer contents.

        Args:
            bufferArray: 2D numpy array. Shape (W, H) where W = horizontal pixels,
                         H = vertical pixels. Will be transformed according to the
                         widget's `transform` setting before display.
        """
        # ── Apply the pre-display transformation ──
        if self._transformName == "log1p":
            displayArray = np.log1p(bufferArray)
        elif self._transformName == "sqrt":
            displayArray = np.sqrt(np.maximum(bufferArray, 0.0))
        else:
            displayArray = bufferArray

        # autoLevels=True: rescale color range to current min/max each frame
        # This is convenient for visualization but loses absolute reference.
        self._imageItem.setImage(displayArray, autoLevels=True)

class MainWindow(QtWidgets.QMainWindow):
    """
    Main application window composing two BufferDisplay widgets side by side.

    The window is a simple assembler: it accepts pre-built BufferDisplay
    instances and lays them out horizontally. It does not know anything about
    what's being displayed.

    Args:
        leftDisplay : BufferDisplay shown on the left.
        rightDisplay: BufferDisplay shown on the right.
        title       : Window title (optional, defaults to "Light Tracer").

    Example:
        vizDisplay    = BufferDisplay(...)  # configured for ray paths
        sensorDisplay = BufferDisplay(...)  # configured for sensor image
        window        = MainWindow(vizDisplay, sensorDisplay, title="Light Tracer")
        window.show()

        for frame in range(N_FRAMES):
            # ... run simulation ...
            window.updateViews(
                leftBuffer  = vizBuffer.numpy(),
                rightBuffer = sensorBuffer.numpy(),
                frameID     = frame,
            )
    """

    def __init__(
        self,
        leftDisplay : BufferDisplay,
        rightDisplay: BufferDisplay,
        title       : str = "Light Tracer",
        parent      = None,
    ):
        super().__init__(parent)

        self._leftDisplay  = leftDisplay
        self._rightDisplay = rightDisplay

        self.setWindowTitle(title)
        self.resize(1600, 900)

        # ── Central widget with a horizontal layout ──
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Use a QSplitter so the user can resize the two views
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(leftDisplay)
        splitter.addWidget(rightDisplay)
        # Equal initial split
        splitter.setSizes([800, 800])

        layout.addWidget(splitter)

        # Store the base title to append the frame counter
        self._baseTitle = title


    def updateViews(
        self,
        leftBuffer : np.ndarray,
        rightBuffer: np.ndarray,
        frameID    : int = 0,
    ):
        """
        Refresh both displays with new buffer contents.

        Args:
            leftBuffer : 2D numpy array for the left display.
            rightBuffer: 2D numpy array for the right display.
            frameID    : current frame number (shown in title for feedback).
        """
        self._leftDisplay.update(leftBuffer)
        self._rightDisplay.update(rightBuffer)

        # Update title with frame counter
        self.setWindowTitle(f"{self._baseTitle} — frame {frameID}")