import os
import json

cmd = 'python train.py --gpu 0 --stage exer --exer_idx -1'
os.system(cmd)

cmd = 'mkdir ../output/model_dump/exer/'
os.system(cmd)

cmd = 'move ../output/model_dump/snapshot_* ../output/model_dump/exer/.'
os.system(cmd)


with open('../data/Sleek/data/exercise_dict.json') as f:
    data = json.load(f)

for i in range(len(data)):
    cmd = 'python train.py --gpu 0 --stage attr --exer_idx ' + str(i)
    os.system(cmd)
    
    cmd = 'mkdir ../output/model_dump/attr/' + str(i)
    os.system(cmd)

    cmd = 'move ../output/model_dump/snapshot_* ../output/model_dump/attr/' + str(i) + '/.'
    os.system(cmd)

