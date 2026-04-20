from PySide6.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget
from PySide6.QtCore    import Qt, Signal


class LabeledSlider(QWidget):
    valueChanged       = Signal(int)    # Index or raw value (int)
    actualValueChanged = Signal(object) # Actual value (from values list or raw int)

    def __init__(self, name, minVal=None, maxVal=None, initVal=None, values=None):
        super().__init__()

        self.name   = name
        self.values = values  # Optional list

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

            self.slider.setMinimum(0)
            self.slider.setMaximum(len(self.values) - 1)

            if initVal is None:
                init_index = 0
            else:
                # si initVal est une valeur → on cherche son index
                if initVal in self.values:
                    init_index = self.values.index(initVal)
                else:
                    raise ValueError("initVal must be in values list")

            self.slider.setValue(init_index)

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
            display = actual
        else:
            actual = value
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