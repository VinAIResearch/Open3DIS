import os

import numpy as np
import torch
import torch.nn.functional as F

# import torch_scatter
from isbnet.util.rle import rle_decode
from scannetv2_inst_eval import ScanNetEval


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
# data_path = '../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_feat'
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_overlap_merge3d_312"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_feat"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_results_debug"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_results_simple_feat_merge"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_results_simple_feat_dc_feat_best"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_results_simple_feat_dc_feat_0.9"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_hier_dc_feat"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_hier_dc_feat_cond_sim"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_results_overlap_simple_hier_dc_feat_cond_sim_merge3d"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_results_overlap_simple_hier_dc_feat_cond_sim_only2d"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_results_overlap_simple_hier_dc_feat_cond_sim_merge3d_debug"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_hier_dc_feat_cond_sim_agg_euclid"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_results_overlap_simple_hier_dc_feat_cond_sim_agg_mask3d_0.9"
data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_hier_dc_feat_cond_sim_agg_odise_debug"


# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_graph3d_simple_hier_dc_feat_cond_sim_agg_opennscene_ensemble"

# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_debug/openscen_fusion_dbscan"
# data_path = "/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir_nospp"
# data_path = "/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir_test"
# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_result_pla_b20"


# data_path = "../Dataset/ScannetV2/ScannetV2_2D_5interval/version_graph_312scenes/final_results_graph_simple_feat_nospp_merge_cas"


# data_path = "/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir_sam_fusion"
# data_path = "/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir_sam_fusion_nospp"
# data_path =  "/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir"
# data_path = "/home/tdngo/Workspace/3dis_ws/OVIR-3D/results/saved_masks_ovir_sam_fusion_no_feat_sim"

pcl_path = "../Dataset/Scannet200/val"
scenes = os.listdir(data_path)

gtsem = []
gtinst = []

res = []
id = 0

# scenes = scenes[::10]
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

    # breakpoint()

    scene_path = os.path.join(data_path, scene)
    pred_mask = torch.load(scene_path)

    # temp = np.array([tensor.numpy() for tensor in pred_mask['ins']])

    # breakpoint()
    masks, category, score = pred_mask["ins"], pred_mask["final_class"], pred_mask["conf"]
    # matrix_nms(torch.tensor(temp), pred_mask['final_class'], pred_mask['conf']) # torch.tensor(temp), pred_mask['final_class'], pred_mask['conf']
    # breakpoint()
    n_mask = category.shape[0]
    tmp = []
    for ind in range(n_mask):
        if isinstance(masks[ind], dict):
            mask = rle_decode(masks[ind])
        else:
            mask = (masks[ind] == 1).numpy().astype(np.uint8)
        # conf = score[ind] #
        conf = 1.0
        final_class = float(category[ind])
        scene_id = scene.replace(".pth", "")
        tmp.append({"scan_id": scene_id, "label_id": final_class + 1, "conf": conf, "pred_mask": mask})

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
    res.append(tmp)

scan_eval.evaluate(res, gtsem, gtinst)
