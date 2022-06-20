import os
import os.path as osp
import numpy as np
import torch
import json
from glob import glob
import matplotlib.pyplot as plt

data_path = osp.join('..', 'data', 'Sleek', 'data')

exer_trainset = {}
exer_testset = {}
attr_trainset = {}
attr_testset = {}

# exercise dict
with open(osp.join(data_path, 'exercise_dict.json')) as f:
    exer_dict = json.load(f)
exer_num = len(exer_dict)

# annotation
annot_path_list = glob(osp.join(data_path, 'publish_3', '*.json')) # D02-5-165.json
for annot_path in annot_path_list:
    if int(annot_path.split('/')[-1].split('-')[0][1:]) > 20:
        continue

    subject_idx = annot_path.split('/')[-1].split('-')[1]
    with open(annot_path) as f:
        annot = json.load(f)
    
    # train on subjects other than '3'
    if subject_idx == '3':
        is_train = False
    else:
        is_train = True
    
    # exercise type and attrs
    exer_name = annot['type_info']['exercise']
    if is_train:
        if exer_name not in exer_trainset:
            exer_trainset[exer_name] = 0
        exer_trainset[exer_name] += 1
        if exer_name not in attr_trainset:
            attr_trainset[exer_name] = {}
    else:
        if exer_name not in exer_testset:
            exer_testset[exer_name] = 0
        exer_testset[exer_name] += 1
        if exer_name not in attr_testset:
            attr_testset[exer_name] = {}

    attrs = annot['type_info']['conditions']
    attr_vector = np.zeros((len(attrs)), dtype=np.float32)
    for attr in attrs:
        attr_name = attr['condition']
        attr_value = float(attr['value'])
        attr_idx = exer_dict[exer_name]['attr_name'].index(attr_name)
        attr_vector[attr_idx] = attr_value
    attr_string = ''
    for value in attr_vector:
        attr_string += str(value)
    
    if is_train:
        if attr_string not in attr_trainset[exer_name]:
            attr_trainset[exer_name][attr_string] = 0
        attr_trainset[exer_name][attr_string] += 1
    else:
        if attr_string not in attr_testset[exer_name]:
            attr_testset[exer_name][attr_string] = 0
        attr_testset[exer_name][attr_string] += 1
  
# exercise normalize and visualize
cnt = 0
label = []
value_trainset = []
for k in exer_trainset:
    cnt += exer_trainset[k]
for k in exer_trainset:
    exer_trainset[k] /= cnt
    label.append(k)
    value_trainset.append(exer_trainset[k])
cnt = 0
value_testset = []
for k in exer_testset:
    cnt += exer_testset[k]
for k in exer_testset:
    exer_testset[k] /= cnt
    value_testset.append(exer_testset[k])

def create_x(t, w, n, d):
    return [t*x + w*n for x in range(d)]
value_a_x = create_x(2, 0.8, 1, len(label))
value_b_x = create_x(2, 0.8, 2, len(label))

ax = plt.subplot()
ax.bar(value_a_x, value_trainset)
ax.bar(value_b_x, value_testset)

middle_x = [(a+b)/2 for (a,b) in zip(value_a_x, value_b_x)]
ax.set_xticks(middle_x)
ax.set_xticklabels(range(len(label)))
plt.legend(['trainset', 'testset'])
plt.savefig('exer_split_dist.png')

# attribute normalize and visualize
for exer_name in attr_trainset.keys():
    plt.close('all')

    cnt = 0
    label = []
    value_trainset = []
    for k in attr_trainset[exer_name]:
        cnt += attr_trainset[exer_name][k]
    for k in attr_trainset[exer_name]:
        attr_trainset[exer_name][k] /= cnt
        label.append(k)
        value_trainset.append(attr_trainset[exer_name][k])
    cnt = 0
    value_testset = []
    for k in attr_testset[exer_name]:
        cnt += attr_testset[exer_name][k]
    for k in attr_testset[exer_name]:
        attr_testset[exer_name][k] /= cnt
        value_testset.append(attr_testset[exer_name][k])

    def create_x(t, w, n, d):
        return [t*x + w*n for x in range(d)]
    value_a_x = create_x(2, 0.8, 1, len(label))
    value_b_x = create_x(2, 0.8, 2, len(label))

    ax = plt.subplot()
    ax.bar(value_a_x, value_trainset)
    ax.bar(value_b_x, value_testset)

    middle_x = [(a+b)/2 for (a,b) in zip(value_a_x, value_b_x)]
    ax.set_xticks(middle_x)
    ax.set_xticklabels(range(len(label)))
    plt.legend(['trainset', 'testset'])
    plt.savefig(exer_name + '_attr_split_dist.png')

