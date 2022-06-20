import json
import os
import os.path as osp

filename = 'D01-3-004.json'
view = 'view2'
joints_name = ('Left Ear', 'Left Eye', 'Right Ear', 'Right Eye', 'Nose', 'Neck', 'Left Shoulder', 'Left Elbow', 'Left Wrist', 'Left Palm', 'Right Shoulder', 'Right Elbow', 'Right Wrist', 'Right Palm', 'Back', 'Waist', 'Left Hip', 'Left Knee', 'Left Ankle', 'Left Foot', 'Right Hip', 'Right Knee', 'Right Ankle', 'Right Foot')

with open(filename, 'r', encoding='UTF-8') as f:
    data = json.load(f)

for frame_idx in range(8):
    # # image
    img_path = data['frames'][frame_idx][view]['img_key']
    print(img_path)
    # # img_path = osp.join('/mnt/disk1/mks0601/Data/Sleek', img_path)
    # img_path = osp.join('../data/Sleek/data/', img_path)
    # cmd = 'copy ' + img_path + ' ./input_imgs/' + str(frame_idx) + '.jpg'
    # os.system(cmd)
    
    # pose
    coord = []
    for joint_name in joints_name:
        x, y = data['frames'][frame_idx][view]['pts'][joint_name]['x'], data['frames'][frame_idx][view]['pts'][joint_name]['y']
        coord.append([x,y]) # x, y
    
    with open('./input_pose/' + str(frame_idx) + '.json', 'w') as f:
        json.dump(coord, f)
