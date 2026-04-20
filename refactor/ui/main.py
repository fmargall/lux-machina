import sys

from PySide6.QtWidgets import QApplication, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Lux Machina @ MANIBVS")

        from pyedsdk import loadSDKLib
        pathToDllFile = r"C:\Users\Utilisateur\Downloads\EDSDKv132010W\EDSDKv132010W\Windows\EDSDK_64\Dll\EDSDK.dll"
        loadSDKLib(pathToDllFile)

        # Note that the EOSCamera import is done after loading SDK library,
        # as it relies on the SDK functions.
        from pyedsdk.camera import EOSCamera
        self.camera = EOSCamera(0)

        from camera_widget import CameraWidget

        self.cameraWidget = CameraWidget(self.camera)
        self.setCentralWidget(self.cameraWidget)

    def closeEvent(self, event):
        try:
            self.camera._close()
        except Exception as e:
            print(f"Error while closing camera: {e}")
        event.accept()


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()