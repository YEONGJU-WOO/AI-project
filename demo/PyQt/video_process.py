import sys
import os.path as osp
import json
import cv2
import torch.backends.cudnn as cudnn
from skeleton_function import transform_joint, TSM_edges, angle_list, center_of_mass_1
import time

sys.path.insert(0, osp.join('../..', 'main'))
sys.path.insert(0, osp.join('../..', 'data'))
sys.path.insert(0, osp.join('../..', 'common'))
sys.path.insert(0, osp.join('..'))
from config import cfg
from model import get_model
from utils.preprocessing import load_video, augmentation, generate_patch_image, process_pose

from utils.vis import vis_keypoints
from SNUEngine import SNUModel
import mediapipe as mp
import numpy as np

from PyQt5.QtCore import *


class exercise(QObject):
    # user_signal = pyqtSignal(bool)
    # time_signal = pyqtSignal(int)
    def __init__(self, webcam_window):
        super().__init__()
        self.image_list = []
        self.keypoints = []
        self.keypoints_3d = []
        self.keypoints_angle = []
        self.center_of_mass = []
        self.recogs = []
        self.tt_num = 10
        self.bb_num = 5



        self.exer_num = 41
        self.joint_num = 24
        with open(osp.join('../exercise_dict_1.json')) as f:
            self.exer_dict = json.load(f)


        # self.user.break_signal.connect()
        # self.user.timer_signal.connect()


        self.TIMER = int(10)
        self.break_time = int(5)
        self.SET = int(5)
        self.prev = time.time()

        self.frames = 0
        self.rep = 0
        self.rep_prev = 0


        self.mp_pose = mp_pose = mp.solutions.pose
        self.pose = mp_pose.Pose()
        self.webcam_window = webcam_window

        self.qtimer = QTimer(self)
        self.qtimer.setInterval(1000)
        self.qtimer.timeout.connect(self.timer_timeout)
        self.timer_start()


    def get_info(self, frame):

        #######################
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #########################

        # Recolor Image to RGB
        h, w = frame.shape[:2]
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detection
        results = self.pose.process(image)
        # Recolor back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



        if self.TIMER != 0 and self.break_time != 0:
            cv2.putText(image, "READY", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5, cv2.LINE_AA)
        elif self.TIMER != 0 and self.break_time ==0:
            try:
                self.image_list.append(frame)
                lm = results.pose_landmarks
                lm_2d = lm.landmark
                lm_3d = results.pose_world_landmarks.landmark

                joint_2d = transform_joint(lm_2d, 2)
                joint_3d = transform_joint(lm_3d, 3)

                joint_2d[:, 0] = joint_2d[:, 0] * w
                joint_2d[:, 1] = joint_2d[:, 1] * h
                joint_2d = joint_2d.astype(int)

                self.keypoints.append(joint_2d)
                self.keypoints_3d.append(joint_3d)

                kp_list, com, com_3d = center_of_mass_1(joint_2d, joint_3d, frame)
                self.center_of_mass.append(com_3d)
                com_x = int(com[0])
                com_y = int(com[1])
                cv2.circle(image, (com_x, com_y), 7, (0, 0, 255), -1)

                ######
                self.keypoints_angle.append(angle_list(joint_3d))
                ######

                for x, y in joint_2d:
                    cv2.circle(image, (x, y), 6, (255, 0, 255), -1)

                for a, b in TSM_edges:
                    cv2.line(image, joint_2d[a], joint_2d[b], (0, 255, 0), 3, 4)

            except:
                pass



        elif self.TIMER == 0 and self.break_time != 0:
            if self.break_time == self.bb_num:
                np.savez(f'data/list_set_{5-self.SET}', image_list = self.image_list,
                         keypoint = self.keypoints, keypoints_3d = self.keypoints_3d,
                         keypoints_angle = self.keypoints_angle, center_of_mass = self.center_of_mass)
            if self.break_time == 1:
                self.image_list = []
                self.keypoints = []
                self.keypoints_3d = []
                self.keypoints_angle = []
                self.center_of_mass = []
            cv2.putText(image, "REST", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 5, cv2.LINE_AA)

        if self.TIMER == 0 and self.break_time == 0:
            print("ending in timer??")

        if self.SET == 0:
            np.savez(f'data/list_set_{5 - self.SET}', image_list=self.image_list,
                     keypoint=self.keypoints, keypoints_3d=self.keypoints_3d,
                     keypoints_angle=self.keypoints_angle, center_of_mass=self.center_of_mass)
            print('DONE!')


        return image

    def timer_timeout(self):

        if self.TIMER == 0 and self.break_time == 0:
            print("ending in timer")
            self.break_time = self.bb_num

        elif self.TIMER != 0 and self.break_time == 0:
            self.TIMER -= 1
            if self.TIMER == 0:
                self.SET -= 1
                self.break_time = self.bb_num

        elif self.TIMER == 0 and self.break_time != 0:
            self.break_time -= 1
            if self.break_time == 0:
                self.TIMER = self.tt_num

        elif self.TIMER != 0 and self.break_time != 0:
            self.break_time -= 1

    def timer_start(self):
        self.qtimer.start()

    def recognition(self):
        cudnn.benchmark = True
        cfg.set_args('0', 'exer', -1)
        model_path = '../../output/model_dump/exer/snapshot_4.pth.tar'
        model = SNUModel(model_path, self.exer_num, self.joint_num)
        for i in range(1,6):
            image_lists = np.load(f'data/list_set_{i}.npz')['image_list']
            keypoints_list = np.load(f'data/list_set_{i}.npz')['keypoint']
            recog = []
            image_list_recog = [image_lists[15*i] for i in range(len(image_lists)//15)]
            keypoints_recog = [keypoints_list[15*i] for i in range(len(keypoints_list)//15)]
            exer_out = model.run_video(image_list_recog, keypoints_recog)
            ind = np.argpartition(exer_out, -5)[-5:]
            n = 5
            for i in ind[np.argsort(exer_out[ind])]:
                for index, k in enumerate(self.exer_dict.keys()):
                    if i == index:
                        # print('%d : ' % (n), k)
                        recog.append(k)
                        n -= 1
            self.recogs.append(recog)