import torch
import numpy as np
import os
import csv
import shutil
from tqdm import tqdm, trange

def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

if os.path.exists('../submission_opensun3d') == False:
    os.mkdir('../submission_opensun3d')
    os.mkdir('../submission_opensun3d/predicted_masks')
scenes = []
with open('open3dis/dataset/opensun3d.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        scenes.append(row)
scenes.pop(0)
tag = [0, 1, 2, 3, 4, 5, 6 ,7, 8, 9, 10, 11, 12, 13, 14, 15 ,16, 17, 18, 19, 20, 21, 22, 23, 24]

for id in trange(len(scenes)):
    scene_id = scenes[id][0]
    class_names = [scenes[id][5]]
    path = '../submission_opensun3d/'+scene_id + '.txt'
    instance_path = '../exp_arkit/version_text/final_result_hier_agglo/'+scene_id+'.pth'
    try:
        inst_result = torch.load(instance_path)
    except:
        print('Empty ', instance_path)
        continue
    instance = inst_result['ins']
    instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])

    confidence = inst_result['conf']
    n_instance = instance.shape[0]
    with open('../submission_opensun3d/' + scene_id + '.txt', 'a') as file:
        cnt = 0
        for ind in range(n_instance):
            mask = np.array(instance[ind] == 1).astype(int)
            score = confidence[ind]
            mask_path = '../submission_opensun3d/predicted_masks/' + scene_id + '_' + str(cnt).zfill(3)
            np.savetxt(mask_path + '.txt',  mask, fmt='%d')
            # file.write(mask_path + '.txt' + ' ' + str(score.item()) + '\n')
            file.write(mask_path.replace('../submission_opensun3d/','') + '.txt' + ' ' + str(1.0) + '\n')
            cnt += 1