from PySide6.QtWidgets import QFrame, QTabWidget, QVBoxLayout, QWidget

class ModePanel(QWidget):
    def __init__(self):
        super().__init__()

        self._setupUI()

    def _setupUI(self):
        layout = QVBoxLayout(self)

        self._tabs = QTabWidget()

        # --------- Tabs ---------
        self._renderTab = QWidget()
        self._optimTab  = QWidget()

        self._tabs.addTab(self._renderTab, "Rendu")
        self._tabs.addTab(self._optimTab, "Optimisation")

        layout.addWidget(self._tabs)

         # - Separator line -
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator)