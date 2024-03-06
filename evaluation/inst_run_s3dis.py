import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter
from isbnet.util.rle import rle_decode
from scannetv2_inst_eval import ScanNetEval


CLASS_LABELS = [
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "chair",
    "table",
    "bookcase",
    "sofa",
    "board",
    # "clutter"
]
# CLASS_LABELS = [
#         # "ceiling",
#         # "floor",
#         # "wall",
#         # "beam",
#         # "column",
#         "window",
#         # "door",
#         "chair",
#         # "table",
#         "bookcase",
#         "sofa",
#         # "board",
#         # "clutter"
#         ]

# CLASS_LABELS = [
#         # "ceiling",
#         "floor",
#         # "wall",
#         # "beam",
#         # "column",
#         "window",
#         "door",
#         "chair",
#         # "table",
#         # "bookcase",
#         "sofa",
#         "board",
#         # "clutter"
#         ]

scan_eval = ScanNetEval(class_labels=CLASS_LABELS, dataset_name="s3dis")
# data_path = '../data/ScannetV2/ScannetV2_2D_5interval/trainval/version1/final_result_openscene_dbscan'
data_path = "../Dataset/s3dis/version_area4/final_result_graph3d_simple_feat_dc_feat_new_feat_debug"
pcl_path = "../Dataset/s3dis/preprocess_notalign"
scenes = os.listdir(data_path)

gtsem = []
gtinst = []
# id = 0
# for scene in scenes:
#     id += 1
#     print(id)
#     gt_path = os.path.join(pcl_path, scene.replace('.pth','') + '_inst_nostuff.pth')
#     loader = torch.load(gt_path)
#     gtsem.append(np.array(loader[2]).astype(np.int32))
#     gtinst.append(np.array(loader[3]).astype(np.int32))

res = []
id = 0
recall_arr = []
for scene in scenes:
    id += 1

    print(id)
    gt_path = os.path.join(pcl_path, scene.replace(".pth", "") + "_inst_nostuff.pth")
    point, _, sem_gt, inst_gt = torch.load(gt_path)

    n_points = point.shape[0]

    if n_points > 1000000:
        stride = 8
    elif n_points >= 600000:
        stride = 6
    elif n_points >= 400000:
        stride = 2
    else:
        stride = 1
    sem_gt = sem_gt[::stride]
    inst_gt = inst_gt[::stride]
    n_points = len(sem_gt)

    if False:
        new_inst_gt = np.zeros_like(inst_gt) - 100
        new_sem_gt = np.zeros_like(sem_gt) - 100
        # for i,c in enumerate([5,7,9,10]):
        for i, c in enumerate([1, 5, 6, 7, 10, 11]):
            new_sem_gt[sem_gt == c] = i
        new_inst_gt[new_sem_gt != -100] = inst_gt[new_sem_gt != -100]
        sem_gt = new_sem_gt
        inst_gt = new_inst_gt

        inst_unique = np.unique(inst_gt)
        new_inst_gt = np.zeros_like(inst_gt) - 100
        count = 0
        for uni in inst_unique:
            if uni == -100:
                continue
            new_inst_gt[inst_gt == uni] = count
            count += 1
        inst_gt = new_inst_gt

    inst_gt[sem_gt == 12] = -100
    sem_gt[sem_gt == 12] = -100

    gtsem.append(np.array(sem_gt).astype(np.int32))
    gtinst.append(np.array(inst_gt).astype(np.int32))

    # print(id)
    scene_path = os.path.join(data_path, scene)
    pred_mask = torch.load(scene_path)
    # masks = [tensor.numpy() for tensor in pred_mask['ins']]
    # category = pred_mask['final_class']
    # score = pred_mask['conf']
    masks, category, score = pred_mask["ins"], pred_mask["final_class"], pred_mask["conf"]

    n_mask = category.shape[0]
    tmp = []
    for ind in range(n_mask):
        if isinstance(masks[ind], dict):
            mask = rle_decode(masks[ind])
        else:
            mask = (masks[ind] == 1).numpy().astype(np.uint8)
        # mask = proposals_pred[ind].astype(np.uint8)
        # conf = score[ind] #
        conf = 1.0
        final_class = float(category[ind])
        scene_id = scene.replace(".pth", "")
        tmp.append({"scan_id": scene_id, "label_id": final_class + 1, "conf": conf, "pred_mask": mask})
    res.append(tmp)
# dic = {'res': res}
# torch.save(dic, 'tpt.pth')

# pred_insts = torch.load('tpt.pth')['res']
scan_eval.evaluate(res, gtsem, gtinst)
