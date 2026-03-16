import sys

import warp as wp

from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    wp.init()

    app = QApplication(sys.argv)
    sys.exit(app.exec())