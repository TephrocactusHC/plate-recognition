import sys
import PySide6
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QFileDialog
from os.path import expanduser
from main import mainmain
import os
dirname    = os.path.dirname(PySide6.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.hello = ["Hallo Welt", "Hei maailma", "Hola Mundo", "Привет мир"]

        self.button = QtWidgets.QPushButton("选择图片")
        self.text_0 = QtWidgets.QLabel("原始图片", alignment=QtCore.Qt.AlignCenter)
        self.text_1 = QtWidgets.QLabel("车牌图片", alignment=QtCore.Qt.AlignCenter)
        self.text_2 = QtWidgets.QLabel("识别结果", alignment=QtCore.Qt.AlignCenter)
        self.image_0 = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.image_1 = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.result = QtWidgets.QLabel("暂无识别结果", alignment=QtCore.Qt.AlignCenter)

        font = self.text_0.font()
        font.setPixelSize(20)
        self.text_0.setFont(font)
        self.text_1.setFont(font)
        self.text_2.setFont(font)
        self.button.setFont(font)
        font.setPixelSize(30)
        self.result.setFont(font)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text_0)
        self.layout.addWidget(self.image_0)
        self.layout.addWidget(self.text_1)
        self.layout.addWidget(self.image_1)
        self.layout.addWidget(self.text_2)
        self.layout.addWidget(self.result)
        self.layout.addWidget(self.button)

        self.button.clicked.connect(self.pick_image)

    @QtCore.Slot()
    def pick_image(self):
        file_path = QFileDialog.getOpenFileName(self, "选择图片", expanduser('~'), "图片文件 (*.png *.jpg *.bmp)")
        if len(file_path[0]) > 0:
            pixmap_0 = QtGui.QPixmap(file_path[0])
            self.image_0.setPixmap(pixmap_0.scaledToHeight(500))
            text_result = mainmain(file_path[0])
            pixmap_1 = QtGui.QPixmap("./messigray.png")
            self.image_1.setPixmap(pixmap_1.scaledToHeight(60))
            self.result.setText(text_result)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.addLibraryPath(os.path.join(os.path.dirname(QtCore.__file__), "plugins"))
    widget = MyWidget()
    widget.setFixedSize(800, 600)
    widget.show()

    sys.exit(app.exec())
