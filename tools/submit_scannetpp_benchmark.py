# PhucNDA rewritten based on Scannetpp toolkit
import numpy as np
import torch
import yaml
import os
import os.path as osp
from munch import Munch

from tqdm import tqdm, trange
import argparse
import multiprocessing as mp
import time
from functools import partial
from util2d.util import rle_decode
from types import SimpleNamespace
import json
from pathlib import Path

def benchmark_rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = ' '.join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle

def write_json(path, data):
    with open(path, "w") as f:
        f.write(json.dumps(data, indent=4))

def get_label_info(semantic_class_list, instance_class_list):
    label_info = SimpleNamespace()
    # all semantic classes
    label_info.all_class_labels = semantic_class_list
    # indices of semantic classes not present in instance class list
    label_info.ignore_classes = [i for i in range(len(label_info.all_class_labels)) if label_info.all_class_labels[i] not in instance_class_list]
    # ids of all semantic classes
    label_info.all_class_ids = list(range(len(label_info.all_class_labels)))
    # ids of instance classes (all semantic classes - ignored classes)
    label_info.class_labels = [label_info.all_class_labels[i] for i in label_info.all_class_ids if i not in label_info.ignore_classes]
    # ids of instance classes
    label_info.valid_class_ids = [i for i in label_info.all_class_ids if i not in label_info.ignore_classes]

    label_info.id_to_label = {}
    label_info.label_to_id = {}

    for i in range(len(label_info.valid_class_ids)):
        # class id -> class name
        label_info.id_to_label[label_info.valid_class_ids[i]] = label_info.class_labels[i]
        # class name -> class id
        label_info.label_to_id[label_info.class_labels[i]] = label_info.valid_class_ids[i]

    return label_info

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser


def main():

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))
    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])


    ### Get Label ###
           
    with open('./data/scanspp/scanpp/scannetpp/metadata/semantic_benchmark/top100.txt') as f:
        semantic_100 = [line.rstrip() for line in f]
    with open('./data/scanspp/scanpp/scannetpp/metadata/semantic_benchmark/top100_instance.txt') as f:
        instance_100 = [line.rstrip() for line in f]


    label_info = get_label_info(semantic_100, instance_100)
    breakpoint()
    for scene_id in tqdm(scene_ids):
        # Directory setup
        inst_predsformat_out_dir = '../submission_scannetpp_benchmark'
        instance_path = '../exp_scannetpp/version_benchmarkinstance_groundedsam/final_result_hier_agglo/'+scene_id+'.pth'
        Path(inst_predsformat_out_dir).mkdir(parents=True, exist_ok=True)
        # 3DIS
        inst_result = torch.load(instance_path)
        instance = inst_result['ins']
        instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        # Class idx
        label = inst_result['final_class']
        
        # create main txt file
        main_txt_file = str(inst_predsformat_out_dir) + '/' + f'{scene_id}.txt'
        # main txt file lines
        main_txt_lines = []
        # create the dir for the instance masks
        inst_masks_dir = str(inst_predsformat_out_dir) + '/'  + 'predicted_masks'
        Path(inst_masks_dir).mkdir(parents=True, exist_ok=True)
        # record results
        for i in range (instance.shape[0]):
            inst_sem_label = label_info.valid_class_ids[label[i]]
            mask_path_relative = f'predicted_masks/{scene_id}_{i:03d}.json'
            main_txt_lines.append(f'{mask_path_relative} {inst_sem_label} 1.0')
            # save the instance mask to a file in the predicted_masks dir
            mask_path = str(inst_predsformat_out_dir) + '/' + str(mask_path_relative)
            write_json(mask_path, benchmark_rle_encode(instance[i]))
        with open(main_txt_file, 'w') as f:
            f.write('\n'.join(main_txt_lines))
            

if __name__ == "__main__":
    main()