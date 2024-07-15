import os
import yaml
import torch
import argparse
import numpy as np
from munch import Munch
from tqdm import tqdm, trange

# Util
from util2d.grounded_sam import Grounded_Sam # Grounded SAM
from util2d.ram_grounded_sam import RAM_Grounded_Sam # RAM Grounded SAM
from util2d.yoloworld_sam import YOLOWorld_SAM # YOLO-World SAM
from util2d.ram_yoloworld_sam import RAM_YOLOWorld_SAM # RAM YOLO-World SAM

from util2d.util import masks_to_rle

from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200 # Scannet200
from open3dis.dataset.scannetpp import INSTANCE_CAT_SCANNET_PP # 1500+ instance classes ScannetPP
from open3dis.dataset.scannetpp import INSTANCE_BENCHMARK84_SCANNET_PP # 84 instance benchmark ScannetPP
from open3dis.dataset.replica import INSTANCE_CAT_REPLICA
from open3dis.dataset.s3dis import INSTANCE_CAT_S3DIS

############################################## Foundations 2D + SAM ##############################################
'''
We generate class-agnostic 2D masks based on {DATASET} class name and CLIP features using 2D foundation models -> Speed
'''


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    if cfg.data.dataset_name == 'scannet200':
        class_names = INSTANCE_CAT_SCANNET_200
    elif cfg.data.dataset_name == 'scannetpp':
        class_names = INSTANCE_CAT_SCANNET_PP    
    elif cfg.data.dataset_name == 'scannetpp_benchmark':
        class_names = INSTANCE_BENCHMARK84_SCANNET_PP
    elif cfg.data.dataset_name == 'replica':
        class_names = INSTANCE_CAT_REPLICA
    elif cfg.data.dataset_name == 's3dis':
        class_names = INSTANCE_CAT_S3DIS
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

    # Fondation model loader
    if cfg.segmenter2d.model == 'Grounded-SAM':
        model = Grounded_Sam(cfg)
    elif cfg.segmenter2d.model == 'RAM Grounded-SAM':
        model = RAM_Grounded_Sam(cfg)
    elif cfg.segmenter2d.model == 'YoloW-SAM':
        model = YOLOWorld_SAM(cfg)
    elif cfg.segmenter2d.model == 'RAM YoloW-SAM':
        model = RAM_YOLOWorld_SAM(cfg)

    # Directory Init
    save_dir = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)
    save_dir_feat = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.grounded_feat_output)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_feat, exist_ok=True)

    # Proces every scene
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            # Tracker
            done = False
            path = scene_id + ".pth"
            with open("tracker_2d.txt", "r") as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    if path in line:
                        done = True
                        break
            if done == True:
                print("existed " + path)
                continue
            # Write append each line
            with open("tracker_2d.txt", "a") as file:
                file.write(path + "\n")
            #####################################
            print("Process", scene_id)
            grounded_data_dict, grounded_features = model.gen_grounded_mask_and_feat(
                scene_id,
                class_names,
                cfg=cfg,
            )

            # Save PC features
            torch.save({"feat": grounded_features}, os.path.join(save_dir_feat, scene_id + ".pth"))
            # Save 2D mask
            torch.save(grounded_data_dict, os.path.join(save_dir, scene_id + ".pth"))
            # Free memory
            torch.cuda.empty_cache()