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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

# data
args = parse_args()
cudnn.benchmark = True
img_path = './input_imgs'
pose_path = './input_pose'
with open(osp.join('exercise_dict_1.json')) as f:
    exer_dict = json.load(f)
exer_num = len(exer_dict)
joint_num = 24


# exercise type prediction
cfg.set_args(args.gpu_ids, 'exer', -1)
model_path = '../output/model_dump/exer/snapshot_4.pth.tar'
model = SNUModel(model_path, exer_num, joint_num)
exer_out = model.run(img_path, pose_path)
exer_idx = np.argmax(exer_out)
for k in exer_dict.keys():
    if exer_dict[k]['exercise_idx'] == exer_idx:
        exer_name = k
        break
print('Exercise type: ' + exer_name)



# # attribute prediction
# cfg.set_args(args.gpu_ids, 'attr', exer_idx)
# attr_num = len(exer_dict[exer_name]['attr_name'])
# model_path = '../output/model_dump/attr/' + str(exer_idx) + '/snapshot_4.pth.tar'
# model = SNUModel(model_path, attr_num, joint_num)
# attr_out = model.run(img_path, pose_path)
# for i in range(len(attr_out)):
#     print(exer_dict[exer_name]['attr_name'][i] + ': ' + str(attr_out[i] > 0.5))

