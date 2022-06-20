import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
from glob import glob
from config import cfg
from utils.preprocessing import get_bbox, process_bbox, load_video, augmentation, process_pose
from utils.vis import vis_keypoints

class Sleek(torch.utils.data.Dataset):
    def __init__(self, data_split):
        self.data_split = data_split
        self.data_path = osp.join('..', 'data', 'Sleek', 'data')
         
        self.joint_num = 24
        self.joints_name = ('Left Ear', 'Left Eye', 'Right Ear', 'Right Eye', 'Nose', 'Neck', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Left Palm', 'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Right Palm', 'Back', 'Waist', 'Left Hip', 'Left Knee', 'Left Ankle', 'Left Foot', 'Right Hip', 'Right Knee', 'Right Ankle', 'Right Foot')
        self.datalist = self.load_data()

    def load_data(self):
        datalist = []
        
        # exercise dict
        with open(osp.join(self.data_path, 'exercise_dict.json')) as f:
            exer_dict = json.load(f)
            # print(exer_dict)
        self.exer_num = len(exer_dict)

        # annotation
        annot_path_list = glob(osp.join(self.data_path, 'publish_3', '*','*.json')) # D02-5-165.json

        for annot in annot_path_list:
            if '3d' in annot:
                annot_path_list.remove(annot)


        # print(annot_path_list)
        for annot_path in annot_path_list:
            # temporally use only day under 20

            # print(annot_path)
            day_idx = int(annot_path.split('\\')[-1].split('-')[0][1:])
            # print(day_idx)

            if day_idx > 20:
                continue

            subject_idx = annot_path.split('\\')[-1].split('-')[1]
            # print(subject_idx)


            with open(annot_path, encoding='UTF-8') as f:
                annot = json.load(f)
                # print(f)
            
            # train on subjects other than '3'
            if self.data_split == 'train':
                if subject_idx == '3':
                    continue
            else:
                if subject_idx != '3':
                    continue
            
            # exercise type and attrs
            exer_name = annot['type_info']['exercise']
            exer_label = exer_dict[exer_name]['exercise_idx']
            attrs = annot['type_info']['conditions']
            attr_label = np.zeros((len(attrs)), dtype=np.float32)
            for attr in attrs:
                attr_name = attr['condition']
                attr_value = attr['value']
                attr_idx = exer_dict[exer_name]['attr_name'].index(attr_name)
                attr_label[attr_idx] = attr_value
            
            # train an exercise-specific model for the attrribute predictio
            if cfg.stage == 'attr':
                if exer_label != cfg.exer_idx:
                    continue
                else:
                    self.attr_num = len(attr_label)

            # for each frame
            data_per_view = {}
            for frame_idx in range(len(annot['frames'])):
                
                # for each view
                for view_idx in annot['frames'][frame_idx].keys():
                    # img path
                    img_path = osp.join(self.data_path, annot['frames'][frame_idx][view_idx]['img_key'])

                    # pose
                    pose = annot['frames'][frame_idx][view_idx]['pts']
                    pose_coord = []
                    for joint_name in self.joints_name:
                        pose_coord.append(np.array([pose[joint_name]['x'], pose[joint_name]['y']], dtype=np.float32))
                    pose_coord = np.stack(pose_coord).reshape(-1,2)
                    
                    if view_idx not in data_per_view:
                        data_per_view[view_idx] = {'img_path': [img_path], 'pose': [pose_coord]}
                    else:
                        data_per_view[view_idx]['img_path'].append(img_path)
                        data_per_view[view_idx]['pose'].append(pose_coord)
            
            
            for view_idx in data_per_view.keys():
                if self.data_split == 'train':
                    start_frame_idx = None
                else:
                    start_frame_idx = 0
                
                # check if files exist
                valid_seq = True
                for img_path in data_per_view[view_idx]['img_path']:
                    if not osp.isfile(img_path):
                        valid_seq = False
                if not valid_seq or len(data_per_view[view_idx]['img_path']) <= cfg.frame_per_seg + 1:
                    continue

                data_dict = {'img_path': data_per_view[view_idx]['img_path'], 'pose_coord': np.stack(data_per_view[view_idx]['pose']), 'exer_label': exer_label, 'attr_label': attr_label, 'start_frame_idx': start_frame_idx}
                datalist.append(data_dict)
        
        if cfg.stage == 'attr' and self.data_split == 'train':
            datalist = datalist * 10
        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, start_frame_idx, exer_label, attr_label = data['img_path'], data['start_frame_idx'], data['exer_label'], data['attr_label']
       
        # video load
        video, video_frame_idxs, original_img_shape = load_video(img_path, start_frame_idx)
        resized_shape = video.shape[1:3] # height, width

        # pose information load
        pose_coords = data['pose_coord'][video_frame_idxs].reshape(-1,self.joint_num,2)
        pose_coords[:,:,0] = pose_coords[:,:,0] / original_img_shape[1] * resized_shape[1]
        pose_coords[:,:,1] = pose_coords[:,:,1] / original_img_shape[0] * resized_shape[0]
        
        """
        # for debug
        for i in range(cfg.frame_per_seg):
            img = video[i,:,:,:]
            coord = pose_coords[i].copy()
            for j in range(self.joint_num):
                cv2.circle(img, (int(coord[j][0]), int(coord[j][1])), radius=3, color=(255,0,0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.imwrite(str(idx) + '_' + str(i) + '.jpg', img)
        """

        # augmentation
        bboxs = []
        for i in range(len(video_frame_idxs)):
            bbox = get_bbox(pose_coords[i], np.ones_like(pose_coords[i,:,0]))
            bbox = process_bbox(bbox)
            bboxs.append(bbox)
        bboxs = np.array(bboxs).reshape(-1,4) # xmin, ymin, width, height
        bboxs[:,2] += bboxs[:,0]; bboxs[:,3] += bboxs[:,1]; # xmin, ymin, xmax, ymax
        xmin = np.min(bboxs[:,0]); ymin = np.min(bboxs[:,1]);
        xmax = np.max(bboxs[:,2]); ymax = np.max(bboxs[:,3]);
        bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin], dtype=np.float32) # xmin, ymin, width, height
        video, img2aug_trans = augmentation(video, bbox)
        video = video.transpose(0,3,1,2).astype(np.float32)/255. # frame_num, channel_dim, height, width
        for i in range(len(video_frame_idxs)):
            pose_coords[i] = process_pose(pose_coords[i], img2aug_trans, self.joint_num, resized_shape)
        
        """
        # for debug
        for i in range(cfg.frame_per_seg):
            img = video[i,::-1,:,:].transpose(1,2,0) * 255
            coord = pose_coords[i].copy()
            coord[:,0] = coord[:,0] / cfg.input_hm_shape[1] * cfg.input_img_shape[1]
            coord[:,1] = coord[:,1] / cfg.input_hm_shape[0] * cfg.input_img_shape[0]
            for j in range(self.joint_num):
                cv2.circle(img, (int(coord[j][0]), int(coord[j][1])), radius=3, color=(255,0,0), thickness=-1, lineType=cv2.LINE_AA)
            cv2.imwrite(str(idx) + '_' + str(i) + '.jpg', img)
        """
        
        if cfg.stage == 'exer':
            inputs = {'video': video, 'pose_coords': pose_coords}
            targets = {'exer_label': exer_label}
            meta_info = {}
        elif cfg.stage == 'attr':
            inputs = {'video': video, 'pose_coords': pose_coords}
            targets = {'attr_label': attr_label}
            meta_info = {}
        return inputs, targets, meta_info

    def evaluate(self, outs):

        print('Evaluation start...')
        annots = self.datalist
        assert len(annots) == len(outs)
        sample_num = len(annots)
 
        accuracy = []
        for n in range(sample_num):
            annot = annots[n]
            out = outs[n]
            img_path = annot['img_path']
            
            if cfg.stage == 'exer':
                if annot['exer_label'] == (np.argmax(out['exer'])):
                    accuracy.append(1)
                else:
                    accuracy.append(0)
            elif cfg.stage == 'attr':
                for class_idx in range(self.attr_num):
                    if annot['attr_label'][class_idx] == (out['attr'][class_idx] > 0.5):
                        accuracy.append(1)
                    else:
                        accuracy.append(0)
        
        print('Accuracy: %.4f' % (sum(accuracy) / len(accuracy) * 100))


