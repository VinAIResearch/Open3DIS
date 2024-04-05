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
from open3dis.dataset.scannet_loader import scaling_mapping
from open3dis.dataset import build_dataset
from open3dis.src.clustering.clustering import process_hierarchical_agglomerative
from open3dis.src.fusion_util import NMS_cuda
from open3dis.src.mapper import PointCloudToImageMapper
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm, trange


def rle_encode_gpu_batch(masks):
    """
    Encode RLE (Run-length-encode) from 1D binary mask.
    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
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
    """
    Decode rle to get binary mask.
    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
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


def refine_grounding_features(
    scene_id, cfg, clip_adapter, clip_preprocess, use_2d_proposals=False, use_3d_proposals=True
):
    """
    Cascade Aggregator
    Refine CLIP pointwise feature from multi scale image crop from 3D proposals
    """

    exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)
    cluster_dict_path = os.path.join(exp_path, cfg.exp.clustering_3d_output, f"{scene_id}.pth")

    pc_features_path = os.path.join(exp_path, cfg.exp.grounded_feat_output, f"{scene_id}.pth")
    pc_refined_features_path = os.path.join(exp_path, cfg.exp.refined_grounded_feat_output, f"{scene_id}.pth")

    ### Set up dataloader
    scene_dir = os.path.join(cfg.data.datapath, scene_id)
    loader = build_dataset(root_path=scene_dir, cfg=cfg)

    img_dim = cfg.data.img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
    )

    data_2d = torch.load(cluster_dict_path)
    if isinstance(data_2d["ins"][0], dict):
        instance_2d = torch.stack([torch.from_numpy(rle_decode(ins)) for ins in data_2d["ins"]], dim=0)
    else:
        instance_2d = data_2d["ins"]

    confidence_2d = torch.tensor(data_2d["conf"])

    ########### Proposal branch selection ###########
    if use_3d_proposals:
        agnostic3d_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path, f"{scene_id}.pth")
        agnostic3d_data = torch.load(agnostic3d_path)
        instance_3d_encoded = np.array(agnostic3d_data["ins"])
        confidence_3d = torch.tensor(agnostic3d_data["conf"])

        n_instance_3d = instance_3d_encoded.shape[0]

        if isinstance(instance_3d_encoded[0], dict):
            instance_3d = torch.stack(
                [torch.from_numpy(rle_decode(in3d)) for in3d in instance_3d_encoded], dim=0
            )
        else:
            instance_3d = torch.stack([torch.tensor(in3d) for in3d in instance_3d_encoded], dim=0)

        intersection = torch.einsum("nc,mc->nm", instance_2d.float(), instance_3d.float())
        # print(intersection.shape, instance.shape, )
        ious = intersection / (instance_2d.sum(1)[:, None] + instance_3d.sum(1)[None, :] - intersection)
        ious_max = torch.max(ious, dim=1)[0]

        valid_mask = torch.ones(instance_2d.shape[0], dtype=torch.bool, device=instance_2d.device)
        valid_mask[ious_max >= cfg.final_instance.iou_overlap] = 0

        instance_2d = instance_2d[valid_mask]
        confidence_2d = confidence_2d[valid_mask]

        instance = torch.cat([instance_2d, instance_3d], dim=0)
        confidence = torch.cat([confidence_2d, confidence_3d], dim=0)
    else:
        instance = instance_2d
        confidence = confidence_2d

    if use_2d_proposals and use_3d_proposals:
        instance = torch.cat([instance_2d, instance_3d], dim=0)
        confidence = torch.cat([confidence_2d, confidence_3d], dim=0)
    elif use_2d_proposals and not use_3d_proposals:
        instance = torch.cat([instance_2d], dim=0)
        confidence = torch.cat([confidence_2d], dim=0)
    else:
        instance = torch.cat([instance_3d], dim=0)
        confidence = torch.cat([confidence_3d], dim=0)
    ########### ########### ########### ###########
    n_instance = instance.shape[0]

    points = loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    n_points = points.shape[0]

    # Use 2D only can load feature obtained from 2D masks else init empty features
    if use_2d_proposals == True:
        pc_features = torch.load(pc_features_path)["feat"].cuda().half()
    else:
        pc_features = torch.zeros_like(torch.load(pc_features_path)["feat"].cuda().half()).cuda()

    cropped_regions = []
    batch_index = []
    confidence_feat = []
    inst_inds = []

    # CLIP point cloud features
    inst_features = torch.zeros((n_instance, 768), dtype=torch.half, device=points.device)

    # H, W = 968, 1296
    interval = cfg.data.img_interval
    mappings = []
    images = []

    for i in trange(0, len(loader), interval):
        frame = loader[i]
        frame_id = frame["frame_id"]  # str

        pose = loader.read_pose(frame["pose_path"])
        depth = loader.read_depth(frame["depth_path"])
        rgb_img = loader.read_image(frame["image_path"])
        rgb_img_dim = rgb_img.shape[:2]

        if "scannetpp" in cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
            depth = cv2.resize(depth, (img_dim[0], img_dim[1]))
            mapping = torch.ones([n_points, 4], dtype=int, device="cuda")
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic = frame["translated_intrinsics"])

        elif "scannet200" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)
            new_mapping = scaling_mapping(
                torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[0], rgb_img_dim[1]
            )
            mapping[:, 1:4] = torch.cat((new_mapping, mapping[:, 3].unsqueeze(1)), dim=1)

        elif "replica" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)

        else:
            raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

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
                confidence_feat.append(confidence[inst])
                inst_inds.append(inst)
                # reproduce from OpenMask3D
                tmpx1 = int(max(0, x1 - (x2 - x1) * kexp * round))
                tmpy1 = int(max(0, y1 - (y2 - y1) * kexp * round))
                tmpx2 = int(min(H - 1, x2 + (x2 - x1) * kexp * round))
                tmpy2 = int(min(W - 1, y2 + (y2 - y1) * kexp * round))
                x1, y1, x2, y2 = tmpx1, tmpy1, tmpx2, tmpy2

    # Batch forwarding CLIP features
    if len(cropped_regions) != 0:
        crops = torch.stack(cropped_regions).cuda()
        img_batches = torch.split(crops, 64, dim=0)
        image_features = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img_batch in img_batches:
                image_feat = clip_adapter.encode_image(img_batch)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat)
        image_features = torch.cat(image_features, dim=0)

    # Point cloud features accumulation
    print("Cascade-Averaging features")
    for count in trange(len(cropped_regions)):
        pc_features[batch_index[count]] += image_features[count] * confidence_feat[count]
        inst_features[inst_inds[count]] += image_features[count] * confidence_feat[count]

    refined_pc_features = F.normalize(pc_features, dim=1, p=2).half().cpu()
    inst_features = F.normalize(inst_features, dim=1, p=2).half().cpu()
    return refined_pc_features, inst_features

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser

if __name__ == "__main__":
    # Multiprocess logger
    # if os.path.exists("tracker_refine.txt") == False:
    #     with open("tracker_refine.txt", "w") as file:
    #         file.write("Processed Scenes .\n")

    args = get_parser().parse_args()
    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    # Fondation model loader
    clip_adapter, _, clip_preprocess = open_clip.create_model_and_transforms(
        cfg.foundation_model.clip_model, pretrained=cfg.foundation_model.clip_checkpoint
    )
    clip_adapter = clip_adapter.cuda()

    # Directory Init
    save_dir_refined_grounded_feat = os.path.join(
        cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.refined_grounded_feat_output
    )
    os.makedirs(save_dir_refined_grounded_feat, exist_ok=True)

    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            # Tracker
            # done = False
            # path = scene_id + ".pth"
            # with open("tracker_refine.txt", "r") as file:
            #     lines = file.readlines()
            #     lines = [line.strip() for line in lines]
            #     for line in lines:
            #         if path in line:
            #             done = True
            #             break
            # if done == True:
            #     print("existed " + path)
            #     continue
            # # Write and append each line
            # with open("tracker_refine.txt", "a") as file:
            #     file.write(path + "\n")
            # print("Process", scene_id)

            refined_pc_features, inst_features = refine_grounding_features(
                scene_id,
                cfg,
                clip_adapter,
                clip_preprocess,
                use_2d_proposals=cfg.proposals.p2d,
                use_3d_proposals=cfg.proposals.p3d,
            )

            # Saving refined features
            torch.save(
                {"feat": refined_pc_features.half(), "inst_feat": inst_features.half()},
                os.path.join(save_dir_refined_grounded_feat, f"{scene_id}.pth"),
            )

            torch.cuda.empty_cache()
