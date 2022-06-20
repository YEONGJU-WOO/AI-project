import json
from glob import glob

exercise_dict = {}
annot_path_list = glob('./publish_3/*/*[!-3d].json')
print(annot_path_list)
exercise_idx = 0
for annot_path in annot_path_list:
    with open(annot_path, encoding='UTF8') as f:
        data = json.load(f)['type_info']
    seq_idx = annot_path.split('/')[-1].split('-')[2][:-5]
    # print(seq_idx)

    if data['exercise'] not in exercise_dict:
        exercise_dict[data['exercise']] = {'exercise_idx': exercise_idx, 'seq_idx': [], 'attr_name': []}
        exercise_idx += 1
    if seq_idx not in exercise_dict[data['exercise']]['seq_idx']:
        exercise_dict[data['exercise']]['seq_idx'].append(seq_idx)

    for condition in data['conditions']:
        attr_name = condition['condition']
        if attr_name not in exercise_dict[data['exercise']]['attr_name']:
            exercise_dict[data['exercise']]['attr_name'].append(attr_name)

with open('exercise_dict.json', 'w') as f:
    json.dump(exercise_dict, f)

            


 
 
