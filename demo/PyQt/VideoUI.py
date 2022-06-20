import cv2
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import math
import time
import pyqtgraph as pg


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, window, user, fileName, cali=False):
        super().__init__()
        self.window = window
        self.user = user
        self.cali = cali
        self._run_flag = True
        self.fileName = fileName

    def run(self):
        arr = np.load(self.fileName)['image_list']
        arr.shape
        for frame in arr:
            frame = frame.astype(np.uint8).copy()
            self.change_pixmap_signal.emit(frame)
            time.sleep(0.03)
            if frame is None:
                print('END')
                break


class VideoWindow(QMainWindow):

    def __init__(self, main_window, workout, cali=False):
        super().__init__()
        self.width = 1400
        self.height = 900
        self.main_window = main_window
        self.workout = workout
        self.cali = cali
        self.score = None
        self.qtimer = QTimer(self)
        self.webcam_thread = None
        self.fileName = ""
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, self.width, self.height)
        self.setWindowTitle("Video_Start!")
        self.setStyleSheet("background-color: #fcf8ec")

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QHBoxLayout(self.centralWidget())

        self.add_webcam_side()
        self.add_feedback_side()

        self.load_img_btn.clicked.connect(self.load_img)
        self.back_btn.clicked.connect(self.to_main_window)

    def openFile(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self, "Open Movie", './data/')
        if self.fileName != '':
            self.webcam_thread = VideoThread(self, self.cali, self.fileName)
            self.webcam_thread.change_pixmap_signal.connect(self.update_image)
            self.webcam_thread.start()
            self.initUI()
            self.graph()


    def add_webcam_side(self):
        self.webcam_layout = QVBoxLayout()
        self.webcam_label = QLabel(self)
        self.webcam_label.setStyleSheet("background-color: black;")

        self.display_width = 640*1.3
        self.display_height = 480*1.3

        self.back_layout = QHBoxLayout()
        self.back_btn = QPushButton("<- Back")
        self.back_btn.setObjectName('back_btn')
        self.back_btn.setStyleSheet('font: bold 30px; color: #fcf8ec; background-color: #456268; border-radius: 5px; '
                                    'padding: 15px;')
        self.empty_label = QLabel(" ")
        self.empty_label2 = QLabel(" ")

        self.back_layout.addWidget(self.back_btn)
        self.back_layout.addWidget(self.empty_label)

        self.webcam_layout.addStretch(1)
        self.webcam_layout.addWidget(self.webcam_label)
        self.webcam_layout.addStretch(1)
        self.webcam_layout.addLayout(self.back_layout)
        self.webcam_layout.addSpacing(30)
        self.layout.addLayout(self.webcam_layout, stretch=3)

    def add_feedback_side(self):
        self.feedback_layout = QVBoxLayout()
        empty_tt = QLabel(" ")
        empty_tt.setContentsMargins(0,0,0,100)
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setBackground('#fcf8ec')

        empty_bb = QLabel(" ")
        empty_bb.setContentsMargins(0, 100, 0, 0)

        self.load_img_btn = QPushButton("LOAD VIDEO")
        self.load_img_btn.setObjectName('load_img_btn')
        self.load_img_btn.setContentsMargins(0, 50, 0, 500)
        self.load_img_btn.resize(50, 100)
        self.load_img_btn.setStyleSheet('font: bold 30px; color: #fcf8ec; background-color: #d49a89; border-radius: 5px; padding: 10px;')

        self.feedback_layout.addWidget(empty_tt)
        self.feedback_layout.addWidget(self.graph_widget)
        self.feedback_layout.addWidget(empty_bb)
        self.feedback_layout.addWidget(self.load_img_btn)
        self.layout.addLayout(self.feedback_layout, stretch=2)

    @pyqtSlot(np.ndarray)
    def update_image(self, img):
        qt_img = self.convert_cv_qt(img)
        self.webcam_label.setPixmap(qt_img)


    def convert_cv_qt(self, img):
        """Convert opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def load_img(self):
        print("load_img")
        self.openFile()


    def to_main_window(self):
        self.main_window.show()
        self.close()


    def update_feedback(self, performance):
        self.feedback_label.setText(str(performance))

    def update_yoga_name(self, yoga_name):
        self.video_label.setText(yoga_name)


    def finished(self, score):
        self.score = score
        self.timer_console.setText("DONE!")
        self.feedback_label.setText(f"You're DONE! your total score is {score}")


    def graph(self):
        # self.graph_widget = pg.GraphicsLayoutWidget()
        print(self.fileName)
        graph = np.load(self.fileName)
        title_1 = 'Angles'
        title_2 = 'Center of Mass'
        angles = graph['keypoints_angle']
        angles = np.transpose(angles)
        com = graph['center_of_mass']
        com = np.transpose(com)

        p = self.graph_widget.addPlot(row=0, col=0, title="#{}".format(title_1))
        for i in angles:
            color = list(np.random.choice(range(256), size=3))
            p.plot(y=i, pen = color, style=Qt.DotLine)
        q = self.graph_widget.addPlot(row=1, col=0, title="#{}".format(title_2))
        for j in com:
            color = list(np.random.choice(range(256), size=3))
            q.plot(y=j, pen = color, style=Qt.DotLine)