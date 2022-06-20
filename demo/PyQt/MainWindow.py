from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import sys
from WebcamUI import WebcamWindow
from VideoUI import VideoWindow


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.width = 1200
        self.height = 900
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, self.width, self.height)
        self.setWindowTitle('Smart Fitness System')
        self.setStyleSheet("background-color: #fcf8ec")

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QVBoxLayout(self.centralWidget())
        self.main_widget.setLayout(self.layout)

        self.add_welcome_msg()
        self.add_names()
        self.add_icons()
        self.add_buttons()


        self.quick_web_btn.clicked.connect(lambda: self.to_quick_web_window(True))
        self.quick_video_btn.clicked.connect(lambda: self.to_quick_video_window(True))

    def add_welcome_msg(self):
        self.msg_sublayout = QHBoxLayout()
        self.welcome_msg = QLabel()
        self.welcome_msg.setText("<h1>Welcome to Smart Pose Trainer!</h1>")
        self.welcome_msg.setStyleSheet("color: #456268; font-size: 25px;")
        self.welcome_msg.setAlignment(Qt.AlignCenter)
        self.welcome_msg.setFixedHeight(100)
        self.msg_sublayout.addWidget(self.welcome_msg)
        self.welcome_msg.resize(1200, 50)

        self.layout.addLayout(self.msg_sublayout)

    def add_names(self):
        self.name_sublayout = QHBoxLayout()

        self.quick_label = QLabel()
        self.quick_label.setText("<h1> Let's Go!!!!!!!!!!!!! </h1>")


        labels = [self.quick_label]
        for label in labels:
            label.setStyleSheet("color: #456268;")
            label.setFixedHeight(40)
            label.setAlignment(Qt.AlignCenter)
            self.name_sublayout.addWidget(label)

        self.name_sublayout.addSpacing(10)
        self.name_sublayout.setContentsMargins(30, 50, 30, 20)
        self.layout.addLayout(self.name_sublayout)

    def add_icons(self):
        self.icon_sublayout = QHBoxLayout()

        self.quick_icon = QLabel(self.centralWidget())
        quick_pixmap = QPixmap("icons/exercise.png")
        self.quick_icon.setContentsMargins(0, 0, 0, 0)

        icons = [self.quick_icon]
        pixmaps = [quick_pixmap]

        for icon, pixmap in zip(icons, pixmaps):
            icon.setPixmap(pixmap)
            icon.resize(pixmap.width(), pixmap.height())
            self.icon_sublayout.addWidget(icon)

        self.icon_sublayout.setContentsMargins(490, 0, 0, 50)
        self.layout.addLayout(self.icon_sublayout)

    def add_buttons(self):
        self.btn_sublayout = QHBoxLayout()

        self.quick_sublayout = QVBoxLayout()
        self.quick_web_btn = QPushButton('Quick Start', self)
        self.quick_video_btn = QPushButton('Evaluation', self)


        sublayouts = [self.quick_sublayout]
        cali_btns = [self.quick_web_btn]
        norm_btns = [self.quick_video_btn]


        for sublayout, cali, norm in zip(sublayouts, cali_btns, norm_btns):
            sublayout.addWidget(cali)
            sublayout.addWidget(norm)
            cali.setStyleSheet("color: #fcf8ec; background-color: #d49a89; border-radius: 3px; font: bold 20px; padding: 10px;")
            norm.setStyleSheet("color: #fcf8ec; background-color: #79a3b1; border-radius: 3px; font: bold 20px; padding: 10px;")
            self.btn_sublayout.addLayout(sublayout)

        self.btn_sublayout.addSpacing(20)
        self.btn_sublayout.setContentsMargins(300, -20, 300, 100)
        self.layout.addLayout(self.btn_sublayout)

    def to_quick_web_window(self, cali=False):
        print("to_quick_web_window")
        self.workout_window = WebcamWindow(self, 'quick', cali)
        self.close()
        self.workout_window.show()

    def to_quick_video_window(self, cali=False):
        print("to_quick_video_window")
        self.workout_window = VideoWindow(self, 'quick', cali)
        self.close()
        self.workout_window.show()


if __name__ == "__main__":
    App = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(App.exec_())
