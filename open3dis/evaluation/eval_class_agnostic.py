import os

import numpy as np
import torch
from isbnet.util.rle import rle_decode
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from open3dis.dataset.scannetpp import SEMANTIC_CAT_SCANNET_PP # ScannetPP
from scannetv2_inst_eval import ScanNetEval
from tqdm import tqdm
import argparse
import yaml
from munch import Munch

def rle_decode(rle):
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except:
        s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    parser.add_argument("--type",type=str,required = True,help="[2D, 3D, 2D_3D]") # raw 3DIS

    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    eval_type= args.type
    
    if cfg.data.dataset_name  == 'scannet200':
        scan_eval = ScanNetEval(class_labels=INSTANCE_CAT_SCANNET_200, use_label = False, dataset_name = 'scannet200')
        pcl_path = cfg.data.gt_pth # groundtruth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass
    if cfg.data.dataset_name  == 'scannetpp':
        # eval instance + 100plus semantic classes
        scan_eval = ScanNetEval(class_labels=SEMANTIC_CAT_SCANNET_PP, use_label = False, dataset_name = 'scannetpp')
        pcl_path = cfg.data.gt_pth # groundtruth
        if eval_type == '2D':
            data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        if eval_type == '3D':
            data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
        if eval_type == '2D_3D':
            pass


    scenes = sorted([s for s in os.listdir(data_path) if s.endswith(".pth")])

    gtsem = []
    gtinst = []
    res = [] #ScannetV2

    for scene in tqdm(scenes):
        scene_path = os.path.join(data_path, scene)
        try: # skipping heavy scenes
            pred_mask = torch.load(scene_path)
        except:
            print('SKIP: ', scene)
            continue
        gt_path = os.path.join(pcl_path, scene)
        loader = torch.load(gt_path)
        sem_gt, inst_gt = loader[2], loader[3]
        gtsem.append(np.array(sem_gt).astype(np.int32))
        gtinst.append(np.array(inst_gt).astype(np.int32))
        masks = pred_mask['ins']

        n_mask = len(masks)
        tmp = []
        for ind in range(n_mask):
            if isinstance(masks[ind], dict):
                mask = rle_decode(masks[ind])
            else:
                try:
                    mask = (masks[ind] == 1).numpy().astype(np.uint8)
                except:
                    mask = (masks[ind] == 1).astype(np.uint8)

            # conf = score[ind] #
            conf = 1.0

            scene_id = scene.replace('.pth', '')
            tmp.append({"scan_id": scene_id, "label_id": 0, "conf": conf, "pred_mask": mask}) # class-agnostic evaluation
        res.append(tmp)

    scan_eval.evaluate(res, gtsem, gtinst)
