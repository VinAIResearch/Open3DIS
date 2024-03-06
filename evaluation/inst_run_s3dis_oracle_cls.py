import numpy as np
import torch
from scannetv2_inst_eval import ScanNetEval
import os
import torch.nn.functional as F
import torch_scatter
from isbnet.util.rle import rle_decode

CLASS_LABELS = ["ceiling",
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

scan_eval = ScanNetEval(class_labels = CLASS_LABELS, dataset_name='s3dis')
# data_path = '../data/ScannetV2/ScannetV2_2D_5interval/trainval/version1/final_result_openscene_dbscan'
data_path = '../Dataset/s3dis/version_area4/final_result_graph3d_simple_feat_dc_feat'
pcl_path = '../Dataset/s3dis/preprocess_notalign'
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
    gt_path = os.path.join(pcl_path, scene.replace('.pth','') + '_inst_nostuff.pth')
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


    inst_gt[sem_gt==12] = -100
    sem_gt[sem_gt==12] = -100

    gtsem.append(np.array(sem_gt).astype(np.int32))
    gtinst.append(np.array(inst_gt).astype(np.int32))

    # print(id)
                # scene_path = os.path.join(data_path, scene)
                # pred_mask = torch.load(scene_path)
                # # masks = [tensor.numpy() for tensor in pred_mask['ins']]
                # # category = pred_mask['final_class']
                # # score = pred_mask['conf']
                # masks, category, score = pred_mask['ins'], pred_mask['final_class'], pred_mask['conf']

    agnostic3d_path = '../Dataset/s3dis/s3dis_area4_cls_agnostic_pretrainfold4'
    agnostic3d_path = os.path.join(agnostic3d_path, scene)
    data = torch.load(agnostic3d_path)
    instance_3d_encoded = np.array(data['ins'])
    confidence_3d = torch.tensor(data['conf']).float()
    n_instance_3d = instance_3d_encoded.shape[0]
    ## ISBNet
    proposals_pred, scores = [], []
    for in3d in range(n_instance_3d):
        proposals_pred.append(torch.tensor(instance_3d_encoded[in3d]))
        scores.append(confidence_3d[in3d].item())
    proposals_pred = torch.stack(proposals_pred, dim=0)
    proposals_pred = proposals_pred[:, ::stride]
    scores = torch.tensor(scores)


    # proposals_pred = torch.stack([torch.from_numpy(rle_decode(m)) for m in masks])
    # proposals_pred = masks

    semantic_labels = torch.from_numpy(sem_gt)
    instance_labels = torch.from_numpy(inst_gt)
    inst_gt_unique = torch.unique(instance_labels)
    instance_cls = []
    instance_labels_onehot = []
    for uni in inst_gt_unique:
        if uni < 0: continue

        ind_ = torch.nonzero(instance_labels == uni).view(-1)
        instance_labels_onehot.append(instance_labels == uni)
        # breakpoint()
        # if len(ind_) == 0:
        #     breakpoint()
        # if len(ind_ > -z)
        instance_cls.append(semantic_labels[ind_[0]])
    instance_cls = torch.tensor(instance_cls)
    instance_labels_onehot = torch.stack(instance_labels_onehot, dim=0)

    # _, instance_labels = torch.unique(instance_labels, return_inverse=True)
    # instance_labels[instance_labels >= 0] += 1
    # instance_labels[instance_labels < 0] = 0  
    # num_ins = instance_labels.max()+1
    # instance_labels_onehot = F.one_hot(instance_labels, num_classes=num_ins) # npoints, ins
    # instance_labels_onehot = instance_labels_onehot[:, 1:] # remove unlabeled
    # instance_labels_onehot = instance_labels_onehot.permute(1,0) # n_ins, npoints
    # instance_labels_onehot = []
    # for uni in inst_gt_unique:
        # if uni < 0: continue

    # breakpoint()

    intersection = torch.einsum("nc,mc->nm", proposals_pred.float(), instance_labels_onehot.float()) #  n_pred, n_ins
    iou_with_gt = intersection / (proposals_pred.float().sum(1)[:, None] + instance_labels_onehot.float().sum(1)[None, :] - intersection)

    assigned_iou, assigned_gt = torch.max(iou_with_gt, dim=1) # n_pred
    try:
        category = instance_cls[assigned_gt]
    except:
        breakpoint()

    valid = (torch.max(iou_with_gt, dim=0)[0] >= 0.5)
    recall_arr.append(valid.sum()/ valid.shape[0])

    # breakpoint()
    # n_mask = len(masks)
    n_mask = category.shape[0]
    tmp = []
    for ind in range(n_mask):
        # if isinstance(masks[ind], dict):
        #     mask = rle_decode(masks[ind])
        # else:
        #     mask = (masks[ind] == 1).numpy().astype(np.uint8)
        # conf = score[ind] #

        mask = proposals_pred[ind].numpy().astype(np.uint8)
        conf = 1.0
        final_class = float(category[ind])
        scene_id = scene.replace('.pth', '')
        tmp.append({'scan_id': scene_id, 'label_id':final_class + 1, 'conf':conf, 'pred_mask':mask})
    res.append(tmp)
# dic = {'res': res}
# torch.save(dic, 'tpt.pth')
# breakpoint()
# pred_insts = torch.load('tpt.pth')['res']
scan_eval.evaluate(res, gtsem, gtinst)