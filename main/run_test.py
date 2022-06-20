import os
import json
import numpy as np
import datetime

with open('../data/Sleek/data/exercise_dict.json') as f:
    data = json.load(f)
exercise_list = [None for _ in range(len(data))]
for k in data.keys():
    exer_idx = data[k]['exercise_idx']
    exercise_list[exer_idx] = k

# run test codes

cmd = 'copy ../output/model_dump/exer/snapshot_4.pth.tar ../output/model_dump/.'
os.system(cmd)

cmd = 'python test.py --gpu 0 --stage exer --exer_idx -1 --test_epoch 4 > ../output/log/test_exer.log'
with open('../output/log/test_exer.log', 'w') as f:
    f.write(cmd)
os.system(cmd)

for i in range(len(data)):
    if os.path.isfile('../output/model_dump/attr/' + str(i) + '/snapshot_4.pth.tar'):
        pass
    else:
        continue

    cmd = 'copy ../output/model_dump/attr/' + str(i) + '/snapshot_4.pth.tar ../output/model_dump/.'
    os.system(cmd)

    cmd = 'python test.py --gpu 0 --stage attr --exer_idx ' + str(i) + ' --test_epoch 4 > ../output/log/test_attr_' + str(i) + '.log'
    with open('../output/log/test_attr_' + str(i) + '.log', 'w') as f:
        f.write(cmd)
    os.system(cmd)

# show test results
print()
print('Exercise type prediction result')
with open('../output/log/test_exer.log') as f:
    result = f.readlines()
    accuracy = float(result[-3].split(':')[1])
    fps = result[-1].split(':')[1].split(' ')[1]
    print('Accuracy (mAP): ' + str(accuracy))
    print('Time (fps): ' + str(fps))

print()
print('Exercise attribute prediction result')
accuracy_list = []
fps_list = []
for i in range(len(data)):
    if os.path.isfile('../output/log/test_attr_' + str(i) + '.log'):
        pass
    else:
        continue

    with open('../output/log/test_attr_' + str(i) + '.log') as f:
        result = f.readlines()
        if len(result) < 5:
            continue
    accuracy = float(result[-3].split(':')[1])
    fps = float(result[-1].split(':')[1].split(' ')[1])
    print('Exercise name: ' + exercise_list[i])
    print('Accuracy (AP): ' + str(accuracy))
    print('Time (fps): ' + str(fps))

    accuracy_list.append(accuracy)
    fps_list.append(fps)

print()
print('Average of all exercises')
print('Accuracy (mAP): ' + str(np.mean(accuracy_list)))
print('Time (fps): ' + str(np.mean(fps_list)))

