import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

# from scannet200 import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
from feature_fusion.utils_torch import (
    SCANNET_COLOR_MAP_200,
    compute_projected_pts,
    compute_projected_pts_torch,
    compute_relation_matrix_self,
    compute_visibility_mask,
    compute_visibility_mask_torch,
    compute_visible_masked_pts,
    compute_visible_masked_pts_torch,
    find_connected_components,
    read_detectron_instances,
    resolve_overlapping_masks,
)

# import torch_scatter
from isbnet.util.rle import rle_decode
from scannetv2_inst_eval import ScanNetEval


HEAD_CATS_SCANNET_200 = [
    "tv stand",
    "curtain",
    "blinds",
    "shower curtain",
    "bookshelf",
    "tv",
    "kitchen cabinet",
    "pillow",
    "lamp",
    "dresser",
    "monitor",
    "object",
    "ceiling",
    "board",
    "stove",
    "closet wall",
    "couch",
    "office chair",
    "kitchen counter",
    "shower",
    "closet",
    "doorframe",
    "sofa chair",
    "mailbox",
    "nightstand",
    "washing machine",
    "picture",
    "book",
    "sink",
    "recycling bin",
    "table",
    "backpack",
    "shower wall",
    "toilet",
    "copier",
    "counter",
    "stool",
    "refrigerator",
    "window",
    "file cabinet",
    "chair",
    "wall",
    "plant",
    "coffee table",
    "stairs",
    "armchair",
    "cabinet",
    "bathroom vanity",
    "bathroom stall",
    "mirror",
    "blackboard",
    "trash can",
    "stair rail",
    "box",
    "towel",
    "door",
    "clothes",
    "whiteboard",
    "bed",
    "floor",
    "bathtub",
    "desk",
    "wardrobe",
    "clothes dryer",
    "radiator",
    "shelf",
]
COMMON_CATS_SCANNET_200 = [
    "cushion",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "toilet paper",
    "printer",
    "blanket",
    "microwave",
    "shoe",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "basket",
    "fan",
    "laptop",
    "person",
    "paper towel dispenser",
    "oven",
    "rack",
    "piano",
    "suitcase",
    "rail",
    "container",
    "telephone",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "seat",
    "column",
    "bicycle",
    "ladder",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "machine",
    "mat",
    "windowsill",
    "bulletin board",
    "fireplace",
    "mini fridge",
    "water cooler",
    "shower door",
    "pillar",
    "ledge",
    "furniture",
    "cart",
    "decoration",
    "closet door",
    "vacuum cleaner",
    "dish rack",
    "range hood",
    "projector screen",
    "divider",
    "bathroom counter",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "bathroom cabinet",
    "structure",
    "storage organizer",
    "potted plant",
    "mattress",
]
TAIL_CATS_SCANNET_200 = [
    "paper",
    "plate",
    "soap dispenser",
    "bucket",
    "clock",
    "guitar",
    "toilet paper holder",
    "speaker",
    "cup",
    "paper towel roll",
    "bar",
    "toaster",
    "ironing board",
    "soap dish",
    "toilet paper dispenser",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "paper cutter",
    "tray",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "sign",
    "projector",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "broom",
    "guitar case",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "laundry detergent",
    "dumbbell",
    "tube",
    "cd case",
    "closet rod",
    "coffee kettle",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "luggage",
]

# CLASS_LABELS = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
CLASS_LABELS = [
    "chair",
    "table",
    "door",
    "couch",
    "cabinet",
    "shelf",
    "desk",
    "office chair",
    "bed",
    "pillow",
    "sink",
    "picture",
    "window",
    "toilet",
    "bookshelf",
    "monitor",
    "curtain",
    "book",
    "armchair",
    "coffee table",
    "box",
    "refrigerator",
    "lamp",
    "kitchen cabinet",
    "towel",
    "clothes",
    "tv",
    "nightstand",
    "counter",
    "dresser",
    "stool",
    "cushion",
    "plant",
    "ceiling",
    "bathtub",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "backpack",
    "toilet paper",
    "printer",
    "tv stand",
    "whiteboard",
    "blanket",
    "shower curtain",
    "trash can",
    "closet",
    "stairs",
    "microwave",
    "stove",
    "shoe",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "board",
    "washing machine",
    "mirror",
    "copier",
    "basket",
    "sofa chair",
    "file cabinet",
    "fan",
    "laptop",
    "shower",
    "paper",
    "person",
    "paper towel dispenser",
    "oven",
    "blinds",
    "rack",
    "plate",
    "blackboard",
    "piano",
    "suitcase",
    "rail",
    "radiator",
    "recycling bin",
    "container",
    "wardrobe",
    "soap dispenser",
    "telephone",
    "bucket",
    "clock",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "clothes dryer",
    "guitar",
    "toilet paper holder",
    "seat",
    "speaker",
    "column",
    "bicycle",
    "ladder",
    "bathroom stall",
    "shower wall",
    "cup",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "paper towel roll",
    "machine",
    "mat",
    "windowsill",
    "bar",
    "toaster",
    "bulletin board",
    "ironing board",
    "fireplace",
    "soap dish",
    "kitchen counter",
    "doorframe",
    "toilet paper dispenser",
    "mini fridge",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "water cooler",
    "paper cutter",
    "tray",
    "shower door",
    "pillar",
    "ledge",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "furniture",
    "cart",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "decoration",
    "sign",
    "projector",
    "closet door",
    "vacuum cleaner",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "dish rack",
    "broom",
    "guitar case",
    "range hood",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "mailbox",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "projector screen",
    "divider",
    "laundry detergent",
    "bathroom counter",
    "object",
    "bathroom vanity",
    "closet wall",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "dumbbell",
    "stair rail",
    "tube",
    "bathroom cabinet",
    "cd case",
    "closet rod",
    "coffee kettle",
    "structure",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "storage organizer",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "potted plant",
    "luggage",
    "mattress",
]
scan_eval = ScanNetEval(class_labels=CLASS_LABELS)

# data_path = '/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir_detic_fusion3'
# data_path = '/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir_sam_fusion'
# data_path = '../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_312'
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_overlap_merge3d_312"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_feat"
data_path = (
    "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_feat_nospp"
)


# data_path = "/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir_sam_fusion"
# data_path = "/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir_sam_fusion_nospp"
# data_path =  "/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir"
pcl_path = "../Dataset/Scannet200/val"
scenes = os.listdir(data_path)

gtsem = []
gtinst = []

res = []
id = 0

recall_arr = []
recall_dict = {
    "head": 0,
    "common": 0,
    "tail": 0,
}

count_dict = {
    "head": 0,
    "common": 0,
    "tail": 0,
}

for scene in scenes:
    # id += 1
    # print(id)
    # if scene.split('.')[0] not in ['scene0011_00', 'scene0015_00', 'scene0030_00', 'scene0153_00', 'scene0207_00', 'scene0300_00', 'scene0496_00', 'scene0575_00', 'scene0685_00', 'scene0686_00', 'scene0700_00']:
    #     continue

    gt_path = os.path.join(pcl_path, scene.replace(".pth", "") + "_inst_nostuff.pth")
    loader = torch.load(gt_path)

    sem_gt, inst_gt = loader[2], loader[3]
    # sem_gt[sem_gt > 2] = 2
    gtsem.append(np.array(sem_gt).astype(np.int32))
    gtinst.append(np.array(inst_gt).astype(np.int32))

    # scene_path = os.path.join(data_path, scene)
    # pred_mask = torch.load(scene_path)

    # # temp = np.array([tensor.numpy() for tensor in pred_mask['ins']])

    # # breakpoint()
    # masks, category, score = pred_mask['ins'], pred_mask['final_class'], pred_mask['conf']
    # matrix_nms(torch.tensor(temp), pred_mask['final_class'], pred_mask['conf']) # torch.tensor(temp), pred_mask['final_class'], pred_mask['conf']
    #
    # # NOTE ISBNET
    # agnostic3d_path = '../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/class_ag_res_200_isbnetfull'
    # agnostic3d_path = os.path.join(agnostic3d_path, scene.split('.')[0] + '.pth')
    # data = torch.load(agnostic3d_path)
    # instance_3d_encoded = np.array(data['ins'])
    # confidence = torch.tensor(data['conf']).double()
    # n_instance_3d = instance_3d_encoded.shape[0]
    # ## ISBNet
    # res, scores = [], []
    # for in3d in range(n_instance_3d):
    #     res.append(torch.tensor(instance_3d_encoded[in3d]))
    #     scores.append(confidence[in3d].item())
    # res = torch.stack(res, dim=0).cuda()
    # scores = torch.tensor(scores).cuda()

    # NOTE our 2d, 3d
    # experiment = "version_graph_312scenes"
    # datapath = "../Dataset/ScannetV2/ScannetV2_2D_5interval/val"

    # exp_path = os.path.join('../Dataset/ScannetV2/ScannetV2_2D_5interval', experiment)

    # graph3d = os.path.join(exp_path, 'graph3d_overlap_simple_hier_dc_feat_cond_sim')
    # data = torch.load(os.path.join(graph3d, scene.split('.')[0]+'.pth'))

    # confidence = torch.tensor(data['conf']).cuda()
    # # breakpoint()
    # if isinstance(data['ins'][0], dict):
    #     instance = torch.stack([torch.from_numpy(rle_decode(ins)) for ins in data['ins']], dim=0).cuda()
    # else:
    #     instance = torch.stack([ins for ins in data['ins']], dim=0).cuda()

    # iou_matrix, _, recall_matrix = compute_relation_matrix_self(instance)
    # adjacency_matrix = ((iou_matrix >= 0.95) | (recall_matrix >= 0.95))

    # #    & (semantic_similarity_matrix >= 0.75)
    # adjacency_matrix = adjacency_matrix | adjacency_matrix.T

    # # merge instances based on the adjacency matrix
    # connected_components = find_connected_components(adjacency_matrix)
    # M = len(connected_components)

    # merged_instance = torch.zeros((M, instance.shape[1]), dtype=torch.int, device=instance.device)
    # merged_confidence = torch.zeros((M), dtype=torch.float, device=instance.device)
    # # merged_feature = torch.zeros((M, feat.shape[1]), dtype=torch.float, device=proposals_pred.device)
    # for i, cluster in enumerate(connected_components):
    #     merged_instance[i] = instance[cluster].sum(0).bool()
    #     merged_confidence[i] = confidence[cluster].mean()

    #     # highest_cluster_ind = cluster[torch.argmax(confidence[cluster])]
    #     # merged_instance[i] = instance[highest_cluster_ind].bool()
    #     # merged_confidence[i] = confidence[highest_cluster_ind]
    #     # merged_feature[i] = feat[cluster].sum(0)
    # instance, confidence = merged_instance, merged_confidence

    # proposals_pred = torch.cat([instance, res],dim=0)
    # NOTE OVIR-3D

    # data = torch.load(f"/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir/{scene.split('.')[0]}.pth")
    # proposals_pred = data['ins'].cuda()
    # prediction_path = os.path.join("/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/fixed_scannet200_results/aligned_scans",
    #                     scene.split('.')[0],
    #                     'detic_output/',
    #                     'imagenet21k-0.3',
    #                     'predictions',
    #                     'proposed_fusion_detic_iou-0.25_recall-0.50_feature-0.75_interval-300.pkl')
    # # prediction_path = '/home/tdngo/Workspace/3dis_ws/OVIR-3D/data/ScannetV2/ScannetV2_2D_5interval/trainval/scene0011_00/detic_output/groundedsam/scene0011_00/predictions/proposed_fusion_detic_iou-0.25_recall-0.50_feature-0.75_interval-300.pkl'
    # with open(prediction_path, 'rb') as fp:
    #     scene_graph = pickle.load(fp)

    # node_ids = list(scene_graph.nodes)
    # instances = torch.zeros((len(node_ids), sem_gt.shape[0]), dtype=torch.bool)
    # # node_ids.sort(key=lambda x: len(scene_graph.nodes[x]['pt_indices']), reverse=True)
    # for i, node_id in enumerate(node_ids):
    #     node = scene_graph.nodes[node_id]
    #     pt_indices = node['pt_indices']
    #     instances[i, pt_indices] = 1
    # proposals_pred = instances.cuda()
    # proposals_pred = torch.stack([torch.from_numpy(rle_decode(m)) for m in masks])
    # proposals_pred = masks

    data = torch.load(
        f"/home/tdngo/Workspace/3dis_ws/Mask3D/results/mask3d_scannet200_topk100/{scene.split('.')[0]}.pth"
    )
    proposals_pred = torch.stack([torch.from_numpy(rle_decode(ins)) for ins in data["ins"]], dim=0).cuda()

    # proposals_pred = data['ins'].cuda()

    semantic_labels = torch.from_numpy(sem_gt)
    instance_labels = torch.from_numpy(inst_gt)
    inst_gt_unique = torch.unique(instance_labels)
    instance_cls = []
    instance_labels_onehot = []
    for uni in inst_gt_unique:
        if uni < 0:
            continue

        ind_ = torch.nonzero(instance_labels == uni).view(-1)
        instance_labels_onehot.append(instance_labels == uni)
        instance_cls.append(semantic_labels[ind_[0]] - 2)
    instance_cls = torch.tensor(instance_cls).cuda()
    instance_labels_onehot = torch.stack(instance_labels_onehot, dim=0).cuda()

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

    intersection = torch.einsum("nc,mc->nm", proposals_pred.float(), instance_labels_onehot.float())  #  n_pred, n_ins
    iou_with_gt = intersection / (
        proposals_pred.float().sum(1)[:, None] + instance_labels_onehot.float().sum(1)[None, :] - intersection
    )

    assigned_iou, assigned_gt = torch.max(iou_with_gt, dim=1)  # n_pred
    try:
        category = instance_cls[assigned_gt]
    except:
        breakpoint()

    valid = torch.max(iou_with_gt, dim=0)[0] >= 0.5
    for i, cls in enumerate(instance_cls):
        if CLASS_LABELS[cls] in HEAD_CATS_SCANNET_200:
            if valid[i]:
                recall_dict["head"] += 1
            count_dict["head"] += 1
        elif CLASS_LABELS[cls] in COMMON_CATS_SCANNET_200:
            if valid[i]:
                recall_dict["common"] += 1
            count_dict["common"] += 1
        elif CLASS_LABELS[cls] in TAIL_CATS_SCANNET_200:
            if valid[i]:
                recall_dict["tail"] += 1
            count_dict["tail"] += 1

    recall_arr.append(valid.sum() / valid.shape[0])
    # breakpoint()

    # n_mask = category.shape[0]
    # tmp = []
    # for ind in range(n_mask):
    #     # if isinstance(masks[ind], dict):
    #     #     mask = rle_decode(masks[ind])
    #     # else:
    #     #     mask = (masks[ind] == 1).numpy().astype(np.uint8)
    #     mask = proposals_pred[ind].numpy().astype(np.uint8)
    #     # conf = score[ind] #
    #     conf = 1.0
    #     final_class = float(category[ind].item())
    #     scene_id = scene.replace('.pth', '')
    #     tmp.append({'scan_id': scene_id, 'label_id':final_class + 1, 'conf':conf, 'pred_mask':mask})

    # n_mask = category.shape[0]
    # tmp = []
    # for ind in range(n_mask):
    #     if isinstance(masks[ind], dict):
    #         mask = rle_decode(masks[ind])
    #     else:
    #         mask = (masks[ind] == 1).numpy().astype(np.uint8)
    #     conf = score[ind] #
    #     conf = 1.0
    #     final_class = float(category[ind])
    #     scene_id = scene.replace('.pth', '')
    #     tmp.append({'scan_id': scene_id, 'label_id':final_class + 1, 'conf':conf, 'pred_mask':mask})

    # tmp.append({'scan_id': scene_id, 'label_id':0 + 1, 'conf':conf, 'pred_mask':mask})
    # res.append(tmp)

recall_arr = torch.tensor(recall_arr)
print("recall", recall_arr.mean())

for k, v in recall_dict.items():
    print(k, v / count_dict[k])

# scan_eval.evaluate(res, gtsem, gtinst)
