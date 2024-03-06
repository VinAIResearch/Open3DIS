import numpy as np
import pandas as pd
import torch

import argparse
import glob
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# Load external constants
from scannet200_constants import VALID_CLASS_IDS_200
from utils import point_indices_from_group, read_plymesh


warnings.filterwarnings("ignore", category=DeprecationWarning)


CLOUD_FILE_PFIX = "_vh_clean_2"
SEGMENTS_FILE_PFIX = ".0.010000.segs.json"
AGGREGATIONS_FILE_PFIX = ".aggregation.json"
CLASS_IDs = VALID_CLASS_IDS_200

NORMALIZED_CKASS_IDS_200 = [-100 for _ in range(1192)]
REVERSE_NORMALIZED_CKASS_IDS_200 = [-100 for _ in range(200)]

count_id = 2
for i, cls_id in enumerate(VALID_CLASS_IDS_200):
    if cls_id == 1:  # wall
        NORMALIZED_CKASS_IDS_200[cls_id] = 0
        REVERSE_NORMALIZED_CKASS_IDS_200[0] = cls_id
    elif cls_id == 3:  # floor
        NORMALIZED_CKASS_IDS_200[cls_id] = 1
        REVERSE_NORMALIZED_CKASS_IDS_200[1] = cls_id
    else:
        NORMALIZED_CKASS_IDS_200[cls_id] = count_id
        REVERSE_NORMALIZED_CKASS_IDS_200[count_id] = cls_id
        count_id += 1

REVERSE_NORMALIZED_CKASS_IDS_200_np = np.array(REVERSE_NORMALIZED_CKASS_IDS_200)
np.save("reverse_norm_ids.npy", REVERSE_NORMALIZED_CKASS_IDS_200_np)

# SAVED_DIR =


def handle_process(scene_path, output_path, labels_pd, train_scenes, val_scenes):

    scene_id = scene_path.split("/")[-1]
    mesh_path = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}.ply")
    segments_file = os.path.join(scene_path, f"{scene_id}{CLOUD_FILE_PFIX}{SEGMENTS_FILE_PFIX}")
    aggregations_file = os.path.join(scene_path, f"{scene_id}{AGGREGATIONS_FILE_PFIX}")
    info_file = os.path.join(scene_path, f"{scene_id}.txt")

    if scene_id in train_scenes:
        output_file = os.path.join(output_path, "train", f"{scene_id}_inst_nostuff.pth")
        split_name = "train"
    elif scene_id in val_scenes:
        output_file = os.path.join(output_path, "val", f"{scene_id}_inst_nostuff.pth")
        split_name = "val"
    else:
        output_file = os.path.join(output_path, "test", f"{scene_id}_inst_nostuff.pth")
        split_name = "test"

    print("Processing: ", scene_id, "in ", split_name)

    # Rotating the mesh to axis aligned
    info_dict = {}
    with open(info_file) as f:
        for line in f:
            (key, val) = line.split(" = ")
            info_dict[key] = np.fromstring(val, sep=" ")

    if "axisAlignment" not in info_dict:
        rot_matrix = np.identity(4)
    else:
        rot_matrix = info_dict["axisAlignment"].reshape(4, 4)

    pointcloud, faces_array = read_plymesh(mesh_path)
    # points = pointcloud[:, :3]
    # colors = pointcloud[:, 3:6]
    # alphas = pointcloud[:, -1]

    # Rotate PC to axis aligned
    r_points = pointcloud[:, :3].transpose()
    r_points = np.append(r_points, np.ones((1, r_points.shape[1])), axis=0)
    r_points = np.dot(rot_matrix, r_points)
    pointcloud = np.append(r_points.transpose()[:, :3], pointcloud[:, 3:], axis=1)

    # Load segments file
    with open(segments_file) as f:
        segments = json.load(f)
        seg_indices = np.array(segments["segIndices"])

    # Load Aggregations file
    with open(aggregations_file) as f:
        aggregation = json.load(f)
        seg_groups = np.array(aggregation["segGroups"])

    # Generate new labels
    labelled_pc = np.zeros((pointcloud.shape[0], 1))
    instance_ids = np.zeros((pointcloud.shape[0], 1))
    for group in seg_groups:
        segment_points, p_inds, label_id = point_indices_from_group(
            pointcloud, seg_indices, group, labels_pd, CLASS_IDs
        )

        # labelled_pc[p_inds] = label_id
        labelled_pc[p_inds] = NORMALIZED_CKASS_IDS_200[label_id]
        instance_ids[p_inds] = group["id"]

    labelled_pc = labelled_pc.astype(int)
    instance_ids = instance_ids.astype(int)

    torch.save(
        (pointcloud[:, :3], pointcloud[:, 3:6] / 127.5 - 1.0, labelled_pc.reshape(-1), instance_ids.reshape(-1)),
        output_file,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Path to the ScanNet dataset containing scene folders")
    parser.add_argument("--output_root", required=True, help="Output path where train/val folders will be located")
    parser.add_argument("--label_map_file", required=True, help="path to scannetv2-labels.combined.tsv")
    parser.add_argument("--num_workers", default=64, type=int, help="The number of parallel workers")
    parser.add_argument(
        "--train_val_splits_path",
        default="../../Tasks/Benchmark",
        help="Where the txt files with the train/val splits live",
    )
    config = parser.parse_args()

    # Load label map
    labels_pd = pd.read_csv(config.label_map_file, sep="\t", header=0)

    # Load train/val splits
    with open(config.train_val_splits_path + "/scannetv2_train.txt") as train_file:
        train_scenes = train_file.read().splitlines()
    with open(config.train_val_splits_path + "/scannetv2_val.txt") as val_file:
        val_scenes = val_file.read().splitlines()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    if not os.path.exists(train_output_dir):
        os.makedirs(train_output_dir)
    val_output_dir = os.path.join(config.output_root, "val")
    if not os.path.exists(val_output_dir):
        os.makedirs(val_output_dir)

    # Load scene paths
    scene_paths = sorted(glob.glob(config.dataset_root + "/*"))

    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    print("Processing scenes...")
    _ = list(
        pool.map(
            handle_process,
            scene_paths,
            repeat(config.output_root),
            repeat(labels_pd),
            repeat(train_scenes),
            repeat(val_scenes),
        )
    )
