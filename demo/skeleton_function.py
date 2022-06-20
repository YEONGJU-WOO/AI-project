import numpy as np
import math as m

TSM_edges = [
        (0, 1),
        (0, 2),
        (2, 4),
        (1, 3),
        (6, 8),
        (8, 10),
        (5, 7),
        (7, 9),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (5, 6),
        (15, 22),
        (16, 23),
        (11, 21),
        (21, 12),
        (20, 21),
        (5, 20),
        (6, 20),
        (17, 6),
        (17, 5),
    ]

joints_name = {
          0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
          5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
          9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
          13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle',
          17: 'neck', 18: 'left_palm', 19: 'right_palm', 20: 'back_spine', 21: 'waist_spine',
          22: 'left_instep', 23: 'right_instep'
    }

angle_columns = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_hip', 'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle', 'neck']

angle_ids = [
    (17, 5, 7), (17, 6, 8), (5, 7, 9), (6, 8, 10),
    (21, 11, 13), (21, 12, 14), (11, 13, 15), (12, 14, 16),
    (13, 15, 22), (14, 16, 23), (0, 17, 20)
]

def skeletons(Pose):
    lmPose_list = []

    for i in range(len(Pose)):
        lmPose_list.append([Pose[i].x, Pose[i].y, Pose[i].z, round(Pose[i].visibility, 2)])

    return lmPose_list


def transform_joint(lm, dim):
    if dim == 2:
        p_list = [[p.x, p.y] for p in lm]
    elif dim == 3:
        p_list = [[p.x, p.y, p.z] for p in lm]
    np_list = np.array(p_list)

    spine_1 = ((np_list[11] + np_list[12]) / 2) * 0.66 + ((np_list[24] + np_list[23]) / 2) * 0.34  # back
    spine_2 = ((np_list[11] + np_list[12]) / 2) * 0.34 + ((np_list[24] + np_list[23]) / 2) * 0.66  # waist
    neck = (np_list[11] + np_list[12]) / 2

    joint_list = np.array([np_list[0], np_list[2], np_list[5], np_list[7], np_list[8], np_list[11], np_list[12],
                           np_list[13], np_list[14], np_list[15], np_list[16], np_list[23], np_list[24],
                           np_list[25], np_list[26], np_list[27], np_list[28], neck, np_list[19], np_list[20],
                           spine_1, spine_2, np_list[31], np_list[32]])

    return joint_list


def findAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    v1 = a-b
    v2 = c-b

    theta = np.arccos(np.sum((v1*v2))/(np.sqrt(np.sum(v1**2))*np.sqrt(np.sum(v2**2))))
    if np.isnan(theta):
        theta = m.pi
    degree = int(180 / m.pi) * theta
    return degree

def angle_list(list_3d):
    angles = []
    np_3d = np.array(list_3d)
    for i in angle_ids:
        angles.append(findAngle(np_3d[:,:3][i[0]],np_3d[:,:3][i[1]],np_3d[:,:3][i[2]]))
    return angles


# def center_of_mass(lm_2d, lm_3d):
#     com_body_2d = 0
#     com_body_3d = 0
#
#     p_list = [[p.x, p.y] for p in lm_2d]
#     p_list_3d = [[p.x, p.y, p.z] for p in lm_3d]
#
#     com_2d = extract_com(p_list)
#     com_3d = extract_com(p_list_3d)
#
#     keypoints_com = [tuple(np.multiply((x[0], x[1]), [frame.shape[1], frame.shape[0]]).astype(int)) for x in com_2d]
#
#     mass_percent = [0.0694, 0.4346, 0.0271, 0.0162, 0.0271, 0.0162, 0.1416, 0.1416, 0.0433, 0.0433]
#     mass_percent_1 = np.array(mass_percent) * 1.0412328196584757
#
#     for i in range(len(mass_percent)):
#         com_body_2d += com_2d[i] * np.array(mass_percent_1)[i]
#         com_body_3d += com_3d[i] * np.array(mass_percent_1)[i]
#
#     return keypoints_com, com_body_2d, com_body_3d


#####################################################################################

def extract_com(point_list):
    np_list = np.array(point_list)

    head = np_list[0]
    trunk = ((np_list[5] + np_list[6]) / 2) * 0.5514 + ((np_list[11] + np_list[12]) / 2) * 0.4486
    upper_arm_r = (np_list[6] * 0.4228) + (np_list[8] * 0.5772)
    fore_arm_r = (np_list[8] * 0.5426) + (np_list[10] * 0.4574)
    upper_arm_l = (np_list[5] * 0.4228) + (np_list[7] * 0.5772)
    fore_arm_l = (np_list[7] * 0.5426) + (np_list[9] * 0.4574)

    thigh_r = (np_list[12] * 0.5905) + (np_list[14] * 0.4095)
    thigh_l = (np_list[11] * 0.5905) + (np_list[13] * 0.4095)
    shank_r = (np_list[14] * 0.5541) + (np_list[16] * 0.4459)
    shank_l = (np_list[13] * 0.5541) + (np_list[15] * 0.4459)

    cm_list = np.array(
        [head, trunk, upper_arm_r, fore_arm_r, upper_arm_l, fore_arm_l, thigh_r, thigh_l, shank_r, shank_l])

    return cm_list


def center_of_mass_1(lm_2d, lm_3d, frame):
    com_body_2d = 0
    com_body_3d = 0

    p_list = [[p[0], p[1]] for p in lm_2d]
    p_list_3d = [[p[0], p[1], p[2]] for p in lm_3d]

    com_2d = extract_com(p_list)
    com_3d = extract_com(p_list_3d)

    keypoints_com = [tuple(np.multiply((x[0], x[1]), [frame.shape[1], frame.shape[0]]).astype(int)) for x in com_2d]

    mass_percent = [0.0694, 0.4346, 0.0271, 0.0162, 0.0271, 0.0162, 0.1416, 0.1416, 0.0433, 0.0433]
    mass_percent_1 = np.array(mass_percent) * 1.0412328196584757

    for i in range(len(mass_percent)):
        com_body_2d += com_2d[i] * np.array(mass_percent_1)[i]
        com_body_3d += com_3d[i] * np.array(mass_percent_1)[i]

    return keypoints_com, com_body_2d, com_body_3d

