import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import math
from video_process import exercise


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, window, user, cali=False):
        super().__init__()
        self.window = window
        self.user = user
        self._run_flag = True
        self.cali = cali

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while self._run_flag:
            ret, img = cap.read()
            img = self.user.get_info(img)
            self.change_pixmap_signal.emit(img)
        cap.release()

    # ADD MOVE TO ANOTHER PAGE!
    def stop(self):
        self._run_flag = False
        self.wait()


class WebcamWindow(QMainWindow):

    def __init__(self, main_window, workout, cali=False):
        super().__init__()
        self.width = 1200  # original 1200 X 900
        self.height = 900
        self.main_window = main_window
        self.workout = workout
        self.workouts = {'quick': 'Quick_Start', 'eval': 'Evaluation' }
        self.cali = cali
        self.tt_num = 10
        self.bb_num = 5
        self.time_left = 10
        self.break_left = 5
        self.set_left = 5

        self.initUI()


    def initUI(self):
        self.setGeometry(100, 100, self.width, self.height)
        self.setWindowTitle(f'{self.workouts[self.workout]}')
        self.setStyleSheet("background-color: #fcf8ec")

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        self.layout = QVBoxLayout(self.centralWidget())

        self.add_webcam_side()
        self.add_middle_side()
        self.add_backbtn_side()

        self.user = self.what_workout()
        # self.webcam_thread = VideoThread(self, self.user, self.cali)
        self.webcam_thread = VideoThread(self, self.user)


        self.webcam_thread.change_pixmap_signal.connect(self.update_image)
        self.webcam_thread.start()
        # self.perf_window = PerformanceWindow(self.main_window, self.workout, self.total_reps)

        # self.user.time_signal.connect(self.time_ss)
        # self.user.time_ss(self.time_left)


        self.stop_btn.clicked.connect(self.closeEvent)
        self.back_btn.clicked.connect(self.to_main_window)
        self.qtimer = QTimer(self)
        self.qtimer.setInterval(1000)
        self.qtimer.timeout.connect(self.timer_timeout)
        self.timer_start()


    def what_workout(self):
        return exercise(self)

    def add_webcam_side(self):
        self.webcam_layout = QHBoxLayout()
        self.webcam_label = QLabel(self)
        self.webcam_label.setStyleSheet("background:red")
        self.display_width = 640*1.3 #original 640 X 480
        self.display_height = 480*1.3

        self.webcam_layout.addWidget(self.webcam_label, alignment=Qt.AlignCenter)
        self.layout.addLayout(self.webcam_layout)

    def add_backbtn_side(self):
        self.back_layout = QHBoxLayout()

        self.back_btn = QPushButton("<- Back")
        self.back_btn.setObjectName('back_btn')
        self.back_btn.setStyleSheet('font: bold 30px; color: #fcf8ec; background-color: #456268; border-radius: 5px; '
                                    'padding: 15px;')
        self.empty_label = QLabel(" ")
        self.empty_label.setFixedWidth(700)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setObjectName('stop_btn')
        self.stop_btn.setStyleSheet(
            'font: bold 30px; color: #fcf8ec; background-color: #d49a89; border-radius: 5px; padding: 15px;')

        self.back_layout.addWidget(self.back_btn)
        self.back_layout.addWidget(self.empty_label)
        self.back_layout.addWidget(self.stop_btn)
        self.layout.addLayout(self.back_layout)

    def add_middle_side(self):
        self.middle_layout = QHBoxLayout()
        self.middle_layout.setContentsMargins(50,0,0,0)


        self.set_label = QLabel('Sets')
        self.set_console = QLabel(f"{6-self.set_left}/5")

        self.timer_label = QLabel("Timer")
        self.timer_console = QLabel()

        time_min = self.time_left // 60
        time_sec = self.time_left % 60
        timer = f"{time_min:02d}:{time_sec:02d}"
        self.timer_console.setText(timer)


        self.break_label = QLabel('Break_Timer')
        self.break_console = QLabel()
        break_min = self.break_left // 60
        break_sec = self.break_left % 60
        break_timer = f"{break_min:02d}:{break_sec:02d}"
        self.break_console.setText(break_timer)



        labels = [self.set_label, self.set_console, self.timer_label,
                  self.timer_console, self.break_label, self.break_console]

        for label in labels:
            label.setStyleSheet('font: bold 30px; color: #456268;')


        self.sets_layout = QVBoxLayout()
        self.sets_layout.addWidget(self.set_label)
        self.sets_layout.addWidget(self.set_console)

        self.timer_layout = QVBoxLayout()
        self.timer_layout.addWidget(self.timer_label)
        self.timer_layout.addWidget(self.timer_console)

        self.break_layout = QVBoxLayout()
        self.break_layout.addWidget(self.break_label)
        self.break_layout.addWidget(self.break_console)


        self.middle_layout.addLayout(self.sets_layout)
        self.middle_layout.addLayout(self.timer_layout)
        self.middle_layout.addLayout(self.break_layout)

        self.layout.addLayout(self.middle_layout)

    @pyqtSlot(bool)
    def timer_slot(self, arg):
        if arg:
            self.timer_start()
        else:
            self.timer_stop()

    @pyqtSlot(int)
    def reps_slot(self, reps):
        if reps == 1:
            self.end()

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

    def closeEvent(self, event):
        self.webcam_thread.stop()
        self.timer_stop()

    def to_main_window(self):
        self.main_window.show()
        self.close()

    def timer_stop(self):
        self.qtimer.stop()

    def timer_start(self):
        self.qtimer.start()
        self.update_timer_ui()

    def update_break_ui(self):
        time_min = self.break_left // 60
        time_sec = self.break_left % 60

        self.break_console.setText(f'{time_min:02d}:{time_sec:02d}')

    def update_set_ui(self):
        self.set_console.setText(f'{6-self.set_left}/5')

    def timer_timeout(self):
        if self.set_left == 0:
            print("ending in set")
            self.end()

        if self.time_left == 0 and self.break_left == 0:
            print("ending in timer")
            self.break_left = self.bb_num
            self.update_break_ui()

        elif self.time_left != 0 and self.break_left == 0:
            self.time_left -= 1
            self.update_timer_ui()
            if self.time_left == 0:
                self.set_left -= 1
                self.update_set_ui()
                self.break_left = self.bb_num
        elif self.time_left == 0 and self.break_left != 0:
            self.break_left -= 1
            self.update_break_ui()
            if self.break_left == 0:
                self.time_left = self.tt_num
        elif self.time_left != 0 and self.break_left != 0:
            self.break_left -= 1
            self.update_break_ui()



    def end(self):
        print("ENDING IN END func")
        self.timer_console.setText("You're DONE!")
        self.webcam_thread.stop()
        self.timer_stop()
        self.user.recognition()
        self.result_screen = Result(self.user.recogs, self)
        # self.close()

    def update_timer_ui(self):
        time_min = self.time_left // 60
        time_sec = self.time_left % 60

        self.timer_console.setText(f'{time_min:02d}:{time_sec:02d}')


    def update_feedback(self, feedback):
        self.feedback_label.setText(feedback)


class Result(QWidget):
    def __init__(self, recogs, webcam_window):
        super().__init__()
        self.recogs = recogs
        self.width = 900
        self.height = 450
        self.webcam_window = webcam_window
        self.initUI()

    def initUI(self):
        self.setGeometry(150, 150, self.width, self.height)
        self.setWindowTitle("Fitness_Score")
        self.setStyleSheet("background-color: #fcf8ec")
        self.grid = QGridLayout()

        self.grid.addWidget(self.result(0), 0, 0)
        self.grid.addWidget(self.result(1), 0, 1)
        self.grid.addWidget(self.result(2), 1, 0)
        self.grid.addWidget(self.result(3), 1, 1)
        self.grid.addWidget(self.result(4), 2, 0)
        self.grid.addWidget(self.save_btn(), 2, 1)

        self.setLayout(self.grid)

        self.save_btn.clicked.connect(self.save)
        self.close_btn.clicked.connect(self.closeEvent)
        self.show()

    def result(self, i):
        self.groupbox = QGroupBox(f'Result {i+1}') # recog

        self.top_1 = QRadioButton(f'1. {self.recogs[i][0]}')
        self.top_2 = QRadioButton(f'2. {self.recogs[i][1]}')
        self.top_3 = QRadioButton(f'3. {self.recogs[i][2]}')
        self.top_4 = QRadioButton(f'4. {self.recogs[i][3]}')
        self.top_5 = QRadioButton(f'5. {self.recogs[i][4]}')

        grids = [self.groupbox, self.top_1, self.top_2,
                 self.top_3, self.top_4, self.top_5]

        for label in grids:
            label.setStyleSheet('font: bold 15px; color: #456268;')

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.top_1)
        self.vbox.addWidget(self.top_2)
        self.vbox.addWidget(self.top_3)
        self.vbox.addWidget(self.top_4)
        self.vbox.addWidget(self.top_5)

        self.groupbox.setLayout(self.vbox)

        return self.groupbox


    def save_btn(self):
        self.groupbox_6 = QGroupBox('Push Buttons')
        self.groupbox_6.setStyleSheet('font: bold 15px; color: #456268;')

        self.save_btn = QPushButton('SAVE')
        self.save_btn.setStyleSheet('font: bold 35px; color: #fcf8ec; background-color: #456268; border-radius: 5px;')
        self.empty = QLabel()
        self.empty.setStyleSheet('font: bold 35px')
        self.empty.setAlignment(Qt.AlignCenter)

        self.close_btn = QPushButton('CLOSE')
        self.close_btn.setStyleSheet('font: bold 35px; color: #fcf8ec; background-color: #d49a89; border-radius: 5px;')

        self.vbox_6 = QVBoxLayout()
        self.vbox_6.addWidget(self.save_btn)
        self.vbox_6.addWidget(self.empty)
        self.vbox_6.addWidget(self.close_btn)
        self.vbox_6.addStretch(1)
        self.groupbox_6.setLayout(self.vbox_6)

        return self.groupbox_6


    def save(self):
        self.empty.setText("SAVE!!")


    def closeEvent(self, event):
        self.close()
        self.webcam_window.to_main_window()
