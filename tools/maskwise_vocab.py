import argparse
import json
import os
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import yaml
from munch import Munch
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from open3dis.dataset.scannetpp import SEMANTIC_CAT_SCANNET_PP, INSTANCE_CAT_SCANNET_PP # ScannetPP
from open3dis.dataset.scannet_loader import ScanNetReader, scaling_mapping
from open3dis.src.clustering.clustering import process_hierarchical_agglomerative
from open3dis.src.fusion_util import NMS_cuda
from open3dis.src.mapper import PointCloudToImageMapper
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm, trange

############### Maskwise Vocab ###############
'''
Fast experiments, for each 3D masks, find topk views, determine the class using CLIP.
Copy from refine_grounding_feat, but discarding instance-aware poinwise feature
'''

def rle_encode_gpu_batch(masks):
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1

        runs[1::2] -= runs[::2]

        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles

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

def maskwise_feature(cfg, scene_id, clip_adapter, clip_preprocess, instance):
    ### Set up dataloader
    scene_dir = os.path.join(cfg.data.datapath, scene_id)
    scannet_loader = ScanNetReader(root_path=scene_dir, cfg=cfg)

    n_instance = instance.shape[0]

    points = scannet_loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    n_points = points.shape[0]

    img_dim = cfg.data.img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, visibility_threshold= 0.15, intrinsics=scannet_loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
    )    

    cropped_regions = []
    batch_index = []
    # confidence_feat = []
    inst_inds = []

    # CLIP point cloud features
    inst_features = torch.zeros((n_instance, 768), dtype=torch.float32, device=points.device)

    # H, W = 968, 1296
    interval = cfg.data.img_interval * 5
    mappings = []
    images = []

    for i in trange(0, len(scannet_loader), interval):
        frame = scannet_loader[i]
        frame_id = frame["frame_id"]  # str

        pose = scannet_loader.read_pose(frame["pose_path"])
        depth = scannet_loader.read_depth(frame["depth_path"])
        rgb_img = scannet_loader.read_image(frame["image_path"])
        rgb_img_dim = rgb_img.shape[:2]

        if "scannetpp" in cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
            depth = cv2.resize(depth, (img_dim[0], img_dim[1]))
            mapping = torch.ones([n_points, 4], dtype=int, device="cuda")
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic = frame["translated_intrinsics"])

        if "scannet200" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)
            new_mapping = scaling_mapping(
                torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[0], rgb_img_dim[1]
            )
            mapping[:, 1:4] = torch.cat((new_mapping, mapping[:, 3].unsqueeze(1)), dim=1)

        if mapping[:, 3].sum() < 100:  # no points corresponds to this image, skip sure
            continue

        mappings.append(mapping.cpu())
        images.append(rgb_img)

    mappings = torch.stack(mappings, dim=0)
    n_views = len(mappings)

    for inst in trange(n_instance):
        # Obtaining top-k views
        conds = (mappings[..., 3] == 1) & (instance[inst] == 1)[None].expand(n_views, -1)  # n_view, n_points
        count_views = conds.sum(1)
        valid_count_views = count_views > 20
        valid_inds = torch.nonzero(valid_count_views).view(-1)
        if len(valid_inds) == 0:
            continue
        topk_counts, topk_views = torch.topk(
            count_views[valid_inds], k=min(cfg.refine_grounding.top_k, len(valid_inds)), largest=True
        )
        topk_views = valid_inds[topk_views]

        # Multiscale image crop from topk views
        for v in topk_views:
            point_inds_ = torch.nonzero((mappings[v][:, 3] == 1) & (instance[inst] == 1)).view(-1)
            projected_points = torch.tensor(mappings[v][point_inds_][:, [1, 2]]).cuda()
            # Calculate the bounding rectangle
            mi = torch.min(projected_points, axis=0)
            ma = torch.max(projected_points, axis=0)
            x1, y1 = mi[0][0].item(), mi[0][1].item()
            x2, y2 = ma[0][0].item(), ma[0][1].item()

            if x2 - x1 == 0 or y2 - y1 == 0:
                continue
            # Multiscale clip crop follows OpenMask3D
            kexp = 0.2
            H, W = images[v].shape[0], images[v].shape[1]
            ## 3 level cropping
            for round in range(3):
                cropped_image = images[v][x1:x2, y1:y2, :]
                if cropped_image.shape[0] == 0 or cropped_image.shape[1] == 0:
                    continue
                cropped_regions.append(clip_preprocess(Image.fromarray(cropped_image)))
                batch_index.append(point_inds_)
                # confidence_feat.append(confidence[inst])
                inst_inds.append(inst)
                # reproduce from OpenMask3D
                tmpx1 = int(max(0, x1 - (x2 - x1) * kexp * round))
                tmpy1 = int(max(0, y1 - (y2 - y1) * kexp * round))
                tmpx2 = int(min(H - 1, x2 + (x2 - x1) * kexp * round))
                tmpy2 = int(min(W - 1, y2 + (y2 - y1) * kexp * round))
                x1, y1, x2, y2 = tmpx1, tmpy1, tmpx2, tmpy2

    # Batch forwarding CLIP features
    if len(cropped_regions) != 0:
        crops = torch.stack(cropped_regions)
        img_batches = torch.split(crops, 64, dim=0)
        image_features = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img_batch in img_batches:
                image_feat = clip_adapter.encode_image(img_batch.cuda())
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat)
        image_features = torch.cat(image_features, dim=0)

    # Point cloud features accumulation
    print("Cascade-Averaging features")
    counter = torch.zeros((inst_features.shape[0], inst_features.shape[1]),dtype=torch.float32, device = points.device)
    for count in trange(len(cropped_regions)):
        inst_features[inst_inds[count]] += image_features[count] # * confidence_feat[count]
        counter[inst_inds[count]] += 1

    counter[counter==0]=1e-5
    inst_features /= counter
    inst_features = inst_features.cpu()
    
    # Maskwise instance features
    return inst_features.cpu()


def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    parser.add_argument("--type",type=str,required = True,help="[2D, 3D, 2D_3D]") # raw 3DIS Choosing which mask set for open-vocab inference

    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))
    
    # Choosing instance branches
    data_path_3d = None
    eval_type = args.type
    if eval_type == '2D':
        data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    if eval_type == '3D':
        data_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)
    if eval_type == '2D_3D':
        data_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
        data_path_3d = os.path.join(cfg.data.cls_agnostic_3d_proposals_path)


    # Loading CLIP
    class_names = ['others']
    clip_adapter, _, clip_preprocess = open_clip.create_model_and_transforms(
    cfg.foundation_model.clip_model, pretrained=cfg.foundation_model.clip_checkpoint)
    clip_adapter = clip_adapter.cuda()
    if cfg.data.dataset_name == 'scannet200':
        class_names = INSTANCE_CAT_SCANNET_200
        if os.path.exists("../pretrains/text_features/scannet200_text_features.pth"):
            text_features = torch.load("../pretrains/text_features/scannet200_text_features.pth").cuda()
        else:
            try:
                os.makedirs('../pretrains/text_features')
            except:
                pass
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = clip_adapter.encode_text(open_clip.tokenize(class_names).cuda())
                text_features /= text_features.norm(dim=-1, keepdim=True)
                torch.save(text_features.cpu(), "../pretrains/text_features/scannet200_text_features.pth")
    if cfg.data.dataset_name == 'scannetpp':
        class_names = INSTANCE_CAT_SCANNET_PP
        if os.path.exists("../pretrains/text_features/scannetpp_text_features.pth"):
            text_features = torch.load("../pretrains/text_features/scannetpp_text_features.pth").cuda()
        else:
            try:
                os.makedirs('../pretrains/text_features')
            except:
                pass
            with torch.no_grad(), torch.cuda.amp.autocast():
                text_features = clip_adapter.encode_text(open_clip.tokenize(class_names).cuda())
                text_features /= text_features.norm(dim=-1, keepdim=True)
                torch.save(text_features.cpu(), "../pretrains/text_features/scannetpp_text_features.pth")   
    
    # Mask-wise features
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])
    # Multiprocess logger
    if os.path.exists("tracker_maskwise.txt") == False:
        with open("tracker_maskwise.txt", "w") as file:
            file.write("Processed Scenes .\n")

    for scene_id in tqdm(scene_ids):
        print("Process", scene_id)
        # Tracker
        done = False
        path = scene_id + ".pth"
        with open("tracker_maskwise.txt", "r") as file:
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
        with open("tracker_maskwise.txt", "a") as file:
            file.write(path + "\n")
        
        # Prep instance
        scene_path = os.path.join(data_path, path)
        pred_mask = torch.load(scene_path)
        masks = pred_mask['ins']
        n_mask = len(masks)
        instance = []
        for ind in range(n_mask):
            if isinstance(masks[ind], dict):
                mask = torch.tensor(rle_decode(masks[ind]))
            instance.append(mask)
        if data_path_3d != None:
            scene_path = os.path.join(data_path_3d, path)
            pred_mask = torch.load(scene_path)
            masks = pred_mask['ins']
            n_mask = len(masks)
            for ind in range(n_mask):
                if isinstance(masks[ind], dict):
                    mask = torchg.tensor(rle_decode(masks[ind]))
                instance.append(mask)
        instance = torch.stack(instance)
        # Run 
        inst_features = maskwise_feature(cfg, scene_id, clip_adapter, clip_preprocess, instance)
        
        # Saving refined features for future offline use :)
        save_dir_refined_grounded_feat = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.refined_grounded_feat_output)
        os.makedirs(save_dir_refined_grounded_feat, exist_ok=True)
        torch.save(
            {"feat": None, "inst_feat": inst_features},
            os.path.join(save_dir_refined_grounded_feat, f"{scene_id}.pth"),)

        predicted_class_score = torch.zeros((inst_features.shape[0], text_features.shape[0]), dtype = torch.float32)
        bs = 100
        for batch in range(0, inst_features.shape[0], bs):
            start = batch
            end = min(start + bs, inst_features.shape[0])
            predicted_class_score[start:end] = (cfg.final_instance.scale_semantic_score * inst_features[start:end].cpu().to(torch.float32) @ text_features.T.cpu().to(torch.float32)).softmax(dim=-1).cpu()
        predicted_class = torch.argmax(predicted_class_score, dim = -1)
        
        inst_class_scores = predicted_class_score.reshape(-1)  # n_cls * n_queries

        cls_final = predicted_class
        scores_final = inst_class_scores
        masks_final = instance
        final_dict = {
            "ins": rle_encode_gpu_batch(masks_final),
            "conf": scores_final.cpu(),
            "final_class": cls_final.cpu(),
        }
        # Final instance
        save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output)
        os.makedirs(save_dir_final, exist_ok=True)
        torch.save(final_dict, os.path.join(save_dir_final, f"{scene_id}.pth"))