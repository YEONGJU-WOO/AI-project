import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import json
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_video, augmentation, generate_patch_image, process_pose
from utils.vis import vis_keypoints
from SNUEngine import SNUModel
from skeleton_function import skeletons, transform_joint, TSM_edges, angle_list, center_of_mass_1
import time
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--gpu', type=str, dest='gpu_ids')
#     args = parser.parse_args()
#
#     # test gpus
#     if not args.gpu_ids:
#         assert 0, print("Please set proper gpu ids")
#
#     if '-' in args.gpu_ids:
#         gpus = args.gpu_ids.split('-')
#         gpus[0] = int(gpus[0])
#         gpus[1] = int(gpus[1]) + 1
#         args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
#
#     return args


# data
# args = parse_args()
cudnn.benchmark = True
with open(osp.join('exercise_dict_1.json')) as f:
    exer_dict = json.load(f)
exer_num = len(exer_dict) #41
joint_num = 24


def video_start():

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    image_list = []
    sk_list = []
    sk_list_3d = []
    sk_com_3d = []
    a_list = []
    TIMER = int(5)

    prev = time.time()


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('END')
            break

        fps = cap.get(cv2.CAP_PROP_FPS)
        # Recolor Image to RGB
        h, w = frame.shape[:2]

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make Detection

        results = pose.process(image)

        #Recolor back to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if TIMER >= 0:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, str(TIMER), (50, 150), font, 6, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.imshow('Mediapipe Feed', image)
            cur = time.time()
            if cur - prev >= 1:
                prev = cur
                TIMER = TIMER - 1

        else:

            try:
                image_list.append(frame)
                lm = results.pose_landmarks
                lm_2d = lm.landmark
                lm_3d = results.pose_world_landmarks.landmark

                joint_2d = transform_joint(lm_2d, 2)
                joint_3d = transform_joint(lm_3d, 3)

                joint_2d[:,0] = joint_2d[:,0]*w
                joint_2d[:,1] = joint_2d[:,1]*h
                joint_2d = joint_2d.astype(int)

                sk_list.append(joint_2d)
                sk_list_3d.append(joint_3d)

                kp_list, com, com_3d = center_of_mass_1(joint_2d, joint_3d, frame)
                sk_com_3d.append(com_3d)
                com_x = int(com[0])
                com_y = int(com[1])
                cv2.circle(image, (com_x, com_y), 7, (0, 0, 255), -1)

                ######
                a_list.append(angle_list(joint_3d))
                ######

                for x, y in joint_2d:
                    cv2.circle(image, (x,y), 6, (255, 0, 255), -1)

                for a,b in TSM_edges:
                    cv2.line(image, joint_2d[a], joint_2d[b], (0, 255, 0), 3, 4)

            except:
                pass

            cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return image_list, sk_list


image_list, sk_list = video_start()

image_list_1 = [image_list[15*i] for i in range(len(image_list)//15)]
sk_list_1 = [sk_list[15*i] for i in range(len(sk_list)//15)]

# exercise type prediction
# cfg.set_args(0, 'exer', -1)
model_path = '../output/model_dump/exer/snapshot_4.pth.tar'
model = SNUModel(model_path, exer_num, joint_num)
# exer_out = model.run_video(image_list, sk_list)
exer_out = model.run_video(image_list_1, sk_list_1)
exer_idx = np.argmax(exer_out)
for k in exer_dict.keys():
    if exer_dict[k]['exercise_idx'] == exer_idx:
        exer_name = k
        break
print('Exercise type: ')

ind = np.argpartition(exer_out, -5)[-5:]
n = 5
for i in ind[np.argsort(exer_out[ind])]:
    for index, k in enumerate(exer_dict.keys()):
        if i == index:
            print('%d : ' % (n), k)
            n -= 1
