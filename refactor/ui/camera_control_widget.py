import math

from PySide6.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget
from PySide6.QtCore    import Qt, Signal


class LabeledSlider(QWidget):
    valueChanged       = Signal(int)    # Index or raw value (int)
    actualValueChanged = Signal(object) # Actual value (from values list or raw int)

    def __init__(self, name, minVal=None, maxVal=None, initVal=None, values=None, displayValues=None):
        super().__init__()

        self.name          = name
        self.values        = values
        self.displayValues = displayValues

        self.label  = QLabel()
        self.slider = QSlider(Qt.Horizontal)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        # Discrete mode (with values list)
        if self.values is not None:
            if len(self.values) == 0:
                raise ValueError("Values list cannot be empty")

            if self.displayValues is not None and len(self.displayValues) != len(self.values):
                raise ValueError("displayValues must have same length as values")

            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.values) - 1)

            if initVal is None:
                initIndex = 0
            else:
                try:
                    initIndex = min(range(len(self.values)), key=lambda i: abs(self.values[i] - initVal))
                except TypeError:
                    initIndex = self.values.index(initVal)

            self.slider.setValue(initIndex)

        # Continuous mode (without values list)
        else:
            if minVal is None or maxVal is None:
                raise ValueError("minVal and maxVal must be provided if values is None")

            self.slider.setMinimum(minVal)
            self.slider.setMaximum(maxVal)
            self.slider.setValue(initVal if initVal is not None else minVal)

        self.slider.valueChanged.connect(self._onValueChanged)

        # Initial update
        self._onValueChanged(self.slider.value())

    def _onValueChanged(self, value):
        if self.values is not None:
            actual = self.values[value]
            if self.displayValues is not None:
                display = self.displayValues[value]
            else:
                display = actual

        else:
            actual  = value
            display = value

        self.label.setText(f"{self.name}: {display}")

        self.valueChanged.emit(value)
        self.actualValueChanged.emit(actual)

    def value(self):
        """Returns the current slider value (index or raw value)"""
        return self.slider.value()

    def actualValue(self):
        """Returns the actual value (useful in list mode)"""
        if self.values is not None:
            return self.values[self.slider.value()]
        return self.slider.value()


class CameraControlWidget(QWidget):
    exposureChanged = Signal(object)
    isoChanged      = Signal(object)
    apertureChanged = Signal(object)

    def __init__(self, camera):
        super().__init__()

        self.camera = camera

        layout = QVBoxLayout()

        # Exposure (shutter speed)
        exposureValues = [
            s.seconds for s in self.camera.availableShutterSpeedList
            if s.seconds is not None
        ]
        exposureDisplay = [
            f"1/{round(1/v)}" if v < 1 else f"{v:.1f}s"
            for v in exposureValues
        ]
        self.exposure = LabeledSlider(
            "Exposure",
            values        = exposureValues,
            displayValues = exposureDisplay,
            initVal       = self.camera.shutterSpeed
        )

        # ISO
        isoValues = [
            s.value for s in self.camera.availableISOList
            if not math.isnan(s.value)
        ]
        isoDisplay = [f"ISO {int(v)}" for v in isoValues]
        self.iso = LabeledSlider(
            "ISO",
            values        = isoValues,
            displayValues = isoDisplay,
            initVal       = self.camera.isoSpeed
        )

        # Aperture
        apertureValues = [
            a.f_number for a in self.camera.availableApertureList
            if a.f_number is not None
        ]
        apertureDisplay = [f"f/{v}" for v in apertureValues]
        self.aperture = LabeledSlider(
            "Aperture",
            values        = apertureValues,
            displayValues = apertureDisplay,
            initVal       = self.camera.aperture
        )

        layout.addWidget(self.exposure)
        layout.addWidget(self.iso)
        layout.addWidget(self.aperture)
        layout.addStretch()

        self.setLayout(layout)

        # Propagating signals
        self.exposure.actualValueChanged.connect(self.exposureChanged)
        self.iso.actualValueChanged.connect(self.isoChanged)
        self.aperture.actualValueChanged.connect(self.apertureChanged)