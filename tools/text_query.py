import argparse
import copy
import json
import multiprocessing as mp
import os
import time
from typing import Dict, Union

import cv2
import groundingdino.datasets.transforms as T
import matplotlib.pyplot as plt
import numpy as np

# CLIP
import open_clip

##### rle_decode
import pycocotools.mask
import torch
import yaml
from munch import Munch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm, trange
from torch.nn import functional as F

# Grounding DINO
from detectron2.structures import BitMasks
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# SAM
from segment_anything import SamPredictor, build_sam, build_sam_hq

from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200 # Scannet200
from open3dis.dataset.scannetpp import SEMANTIC_CAT_SCANNET_PP # ScannetPP
from open3dis.dataset.replica import INSTANCE_CAT_REPLICA
from open3dis.dataset.s3dis import INSTANCE_CAT_S3DIS
from open3dis.dataset.scannet_loader import ScanNetReader, scaling_mapping
from open3dis.dataset import build_dataset

#### Open3DIS util
from open3dis.src.fusion_util import NMS_cuda
from open3dis.src.mapper import PointCloudToImageMapper
from open3dis.src.clustering.clustering import process_hierarchical_agglomerative

# Visualize
import random
import pyviz3d.visualizer as viz
import open3d as o3d

############# UTIL###############
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

def generate_palette(n):
    palette = []
    for _ in range(n):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        palette.append((red, green, blue))
    return palette

def read_pointcloud(pcd_path):
    scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
    point = np.array(scene_pcd.points)
    color = np.array(scene_pcd.colors)

    return point, color

############################################## Grounding DINO + SAM ##############################################

def load_image(image_pil):
    """
    Grounding DINO preprocess
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            # T.RandomResize([400], max_size=400), # not enough memory ? Consider this
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    """
    Grounding DINO loader
    """
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    model.cuda()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    """
    Grounding DINO box generator
    Returning boxes and logits scores for each chunk in the caption with box & text threshoding
    """

    # Caption formatting
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    model = model.to(device)
    image = image.to(device)

    # Grounding DINO box generator
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nqueries, 256)
    boxes = outputs["pred_boxes"][0]  # (nqueries, 4)

    # Filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    return boxes_filt, logits_filt.max(dim=1)[0]


def show_mask(mask, ax, random_color=False):
    """
    Mask visualization
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def masks_to_rle(masks) -> Dict:
    """
    Encode 2D mask to RLE (save memory and fast)
    """
    res = []
    if masks == None:
        return None
    masks = masks.squeeze(1)
    for mask in masks:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        assert isinstance(mask, np.ndarray)
        rle = pycocotools.mask.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        res.append(rle)
    return res


def init_foundation_models(cfg):
    """
    Init foundation model
    """
    # CLIP
    clip_adapter, _, clip_preprocess = open_clip.create_model_and_transforms(
        cfg.foundation_model.clip_model, pretrained=cfg.foundation_model.clip_checkpoint
    )
    clip_adapter = clip_adapter.cuda()
    # Grounding DINO
    grounding_dino_model = load_model(
        cfg.foundation_model.grounded_config_file, cfg.foundation_model.grounded_checkpoint, device="cuda"
    )
    # Segment Anything
    sam_predictor = SamPredictor(build_sam_hq(checkpoint=cfg.foundation_model.sam_checkpoint).to("cuda"))

    return clip_adapter, clip_preprocess, grounding_dino_model, sam_predictor


def gen_grounded_mask_and_feat(
    scene_id, clip_adapter, clip_preprocess, grounding_dino_model, sam_predictor, class_names, cfg, gen_feat=True
):
    """
    Grounding DINO + SAM, CLIP
        Generate 2D masks from GDino box prompt
        Accmulate CLIP mask feature onto 3D point cloud
    Returning boxes and logits scores for each chunk in the caption with box & text threshoding
    """
    scene_dir = os.path.join(cfg.data.datapath, scene_id)

    loader = build_dataset(root_path=scene_dir, cfg=cfg)
    # scannet_loader = ScanNetReader(root_path=scene_dir, cfg=cfg)

    # Pointcloud Image mapper
    img_dim = cfg.data.img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
    )

    points = loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    n_points = points.shape[0]

    grounded_data_dict = {}

    # Accmulate CLIP mask feature onto 3D point cloud ?
    if gen_feat:
        grounded_features = torch.zeros((n_points, cfg.foundation_model.clip_dim)).cuda()
    else:
        grounded_features = None

    for i in trange(0, len(loader), cfg.data.img_interval):
        frame = loader[i]
        frame_id = frame["frame_id"]  # str
        image_path = frame["image_path"]  # str

        #### Processing Grounding DINO ####
        image_pil = Image.open(image_path).convert("RGB")
        image_pil, image_infer = load_image(image_pil)
        boxes_filt = []
        confs_filt = []

        ### Cannot query directly 200 classes so split them into multiple chunks -- see Supplementary
        segment_size = 10
        segments = [class_names[i : i + segment_size] for i in range(0, len(class_names), segment_size)]
        for cls_name in segments:
            boxes, confidences = get_grounding_output(
                grounding_dino_model,
                image_infer,
                ".".join(cls_name),
                cfg.foundation_model.box_threshold,
                cfg.foundation_model.text_threshold,
                device=cfg.foundation_model.device,
            )

            if len(boxes) > 0:
                boxes_filt.append(boxes)
                confs_filt.append(confidences)

        if len(boxes_filt) == 0:  # No box in that view
            continue
        boxes_filt = torch.cat(boxes_filt)
        confs_filt = torch.cat(confs_filt)

        size = image_pil.size
        H, W = size[1], size[0]
        boxes_filt = boxes_filt * torch.Tensor([W, H, W, H])[None, ...].cuda()

        # XYWH to XYXY
        boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
        boxes_filt[:, 2:] += boxes_filt[:, :2]
        l, t, r, b = (
            boxes_filt[:, 0].clip(0),
            boxes_filt[:, 1].clip(0),
            boxes_filt[:, 2].clip(min=0, max=W),
            boxes_filt[:, 3].clip(min=0, max=H),
        )

        # Filtering big boxes
        valid_boxes = ((b - t) > 1) & ((r - l) > 1) & ((b - t) * (r - l) / (W * H) < 0.85)
        target_id_valid = torch.nonzero(valid_boxes).view(-1)

        if len(target_id_valid) == 0:  # No valid box
            continue
        boxes_filt = boxes_filt[target_id_valid]
        confs_filt = confs_filt[target_id_valid]

        # BOX NMS
        boxes_filt, confs_filt = NMS_cuda(boxes_filt, confs_filt, 0.5)
        boxes_filt = torch.stack(boxes_filt)
        confs_filt = torch.tensor(confs_filt)

        #### Segment Anything ####
        image_sam = cv2.imread(image_path)
        image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
        rgb_img_dim = image_sam.shape[:2]
        sam_predictor.set_image(image_sam)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image_sam.shape[:2])  # .to(device)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(cfg.foundation_model.device),
            multimask_output=False,
        )

        if masks == None:  # No mask in the view
            continue

        masks_fitted = torch.zeros_like(masks, dtype=bool)
        regions = []

        for box_id, box in enumerate(boxes_filt):
            l, t, r, b = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
            l = max(l, 0)
            t = max(t, 0)
            r = min(r, W)
            b = min(b, H)
            # Outercrop 2D mask
            masks_fitted[box_id, 0, t:b, l:r] = True
            row, col = torch.where(masks[box_id][0, t:b, l:r] == False)
            tmp = torch.tensor(image_sam)[t:b, l:r, :].cuda()
            # Blurring background - trick here improve CLIP feature
            tmp[row, col, 0] = (0 * 0.5 + tmp[row, col, 0] * (1 - 0.5)).to(torch.uint8)
            tmp[row, col, 1] = (0 * 0.5 + tmp[row, col, 1] * (1 - 0.5)).to(torch.uint8)
            tmp[row, col, 2] = (0 * 0.5 + tmp[row, col, 2] * (1 - 0.5)).to(torch.uint8)
            regions.append(clip_preprocess(Image.fromarray((tmp.cpu().numpy()))))

        masks = torch.logical_and(masks, masks_fitted)  # fitting
        imgs = torch.stack(regions).cuda()
        img_batches = torch.split(imgs, 64, dim=0)
        image_features = []

        # Batch forwarding CLIP
        with torch.no_grad(), torch.cuda.amp.autocast():
            for img_batch in img_batches:
                image_feat = clip_adapter.encode_image(img_batch)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features.append(image_feat)
        image_features = torch.cat(image_features, dim=0)

        if False:
            # draw output image
            image = loader.read_image(image_path)
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            plt.axis("off")
            # plot out
            os.makedirs("./debug/s3dis/" + scene_id, exist_ok=True)
            plt.savefig(
                os.path.join("./debug/s3dis/" + scene_id + "/sam_" + str(i) + ".jpg"),
                bbox_inches="tight",
                dpi=300,
                pad_inches=0.0,
            )

        #### SAVING MASKS, CLIP FEATURES
        grounded_data_dict[frame_id] = {
            "masks": masks_to_rle(masks),
            "img_feat": image_features.cpu(),
            "conf": confs_filt.cpu(),
        }

        if gen_feat:
            pose = loader.read_pose(frame["pose_path"])
            depth = loader.read_depth(frame["depth_path"])
            
            if "scannetpp" in cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
                depth = cv2.resize(depth, (img_dim[0], img_dim[1]))
                mapping = torch.ones([n_points, 4], dtype=int, device="cuda")
                mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic=frame["translated_intrinsics"])

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

            elif "s3dis" in cfg.data.dataset_name:
                mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
                mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic=frame["intrinsics"])

            else:
                raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

            idx = torch.where(mapping[:, 3] == 1)[0]

            if len(idx) < 100:  # No points corresponds to this image, visible points on 2D image
                continue

            pred_masks = BitMasks(masks.squeeze(1))
            # Flood fill single CLIP feature for 2D mask
            final_feat = torch.einsum("qc,qhw->chw", image_features, pred_masks.tensor.float())
            ### Summing features
            grounded_features[idx] += final_feat[:, mapping[idx, 1], mapping[idx, 2]].permute(1, 0)

    grounded_features = grounded_features.cpu()
    return grounded_data_dict, grounded_features


def refine_grounding_features(
    scene_id, cfg, clip_adapter, clip_preprocess):
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
    instance = instance_2d
    confidence = confidence_2d

    n_instance = instance.shape[0]

    points = loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    n_points = points.shape[0]

    # Use 2D only can load feature obtained from 2D masks else init empty features
    pc_features = torch.load(pc_features_path)["feat"].cuda().half()

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

    if cfg.data.dataset_name == 's3dis':
        target_frame = 300
        interval = max(interval, len(loader) // target_frame)

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
        elif "s3dis" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic=frame["intrinsics"])
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
    parser.add_argument("--text_query",type=str,required = True,help="Text_query")

    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()


    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))
    text_query = args.text_query

    # Fondation model loader
    clip_adapter, clip_preprocess, grounding_dino_model, sam_predictor = init_foundation_models(cfg)

    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    class_names = text_query.split('.')
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_adapter.encode_text(open_clip.tokenize(class_names + ['others']).cuda()) # Engineering OpenScene
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Directory Init
    save_dir = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output)
    save_dir_feat = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.grounded_feat_output)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_feat, exist_ok=True)

    # Proces every scene
    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            print("Process", scene_id)
            # NOTE ############################################# Gen 2D masks
            
            print('Gen 2D masks')
            grounded_data_dict, grounded_features = gen_grounded_mask_and_feat(
                scene_id,
                clip_adapter,
                clip_preprocess,
                grounding_dino_model,
                sam_predictor,
                class_names=class_names,
                cfg=cfg,
            )

            # Save PC features
            torch.save({"feat": grounded_features.half()}, os.path.join(save_dir_feat, scene_id + ".pth"))
            # Save 2D mask
            torch.save(grounded_data_dict, os.path.join(save_dir, scene_id + ".pth"))
            torch.cuda.empty_cache()
            # NOTE ############################################# Lift 2D -> 3D
            
            print('Lift 2D -> 3D')
            save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
            os.makedirs(save_dir_cluster, exist_ok=True)

            cluster_dict = None
            proposals3d, confidence = process_hierarchical_agglomerative(scene_id, cfg)
            if proposals3d == None: # Discarding too large scene
                continue
            cluster_dict = {
                "ins": rle_encode_gpu_batch(proposals3d),
                "conf": confidence,
            }
            torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))
            # NOTE ############################################# Feature refinement
            
            print('Feature refinement')
            save_dir_refined_grounded_feat = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.refined_grounded_feat_output)
            os.makedirs(save_dir_refined_grounded_feat, exist_ok=True)

            refined_pc_features, inst_features = refine_grounding_features(
                scene_id,
                cfg,
                clip_adapter,
                clip_preprocess
            )
            # Saving refined features
            torch.save(
                {"feat": refined_pc_features.half(), "inst_feat": inst_features.half()},
                os.path.join(save_dir_refined_grounded_feat, f"{scene_id}.pth"),)
            torch.cuda.empty_cache()
            # NOTE ############################################# Finalizing output
            
            print('Final output')
            save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output) # final_output
            os.makedirs(save_dir_final, exist_ok=True)

            exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)
            # Stage 1
            pc_features_path = os.path.join(exp_path, cfg.exp.grounded_feat_output, f"{scene_id}.pth") 
            # Stage 2
            pc_refined_features_path = os.path.join(exp_path, cfg.exp.refined_grounded_feat_output, f"{scene_id}.pth")
            
            # 2D branch
            save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
            data = torch.load(os.path.join(save_dir_cluster, f"{scene_id}.pth"))
            if isinstance(data["ins"][0], dict):
                instance_2d = torch.stack([torch.from_numpy(rle_decode(ins)) for ins in data["ins"]], dim=0).cuda()
            else:
                instance_2d = data["ins"].cuda()
            confidence_2d = torch.tensor(data["conf"]).cuda()
            
            instance = instance_2d
        
            # Choose your feature stage
            pc_features = torch.load(pc_refined_features_path)["feat"].cuda().half()
            predicted_class = (cfg.final_instance.scale_semantic_score * pc_features.half() @ text_features.cuda().T).softmax(dim=-1)
            inst_class_scores = torch.einsum("kn,nc->kc", instance.float(), predicted_class.float()).cuda()
            inst_class_scores = inst_class_scores / instance.float().cuda().sum(dim=1)[:, None]  # K x classes
            score, cls_final = torch.max(inst_class_scores, dim=-1)
            
            # Thresholding Filter
            threshoding = torch.logical_and((cls_final < len(class_names)), (score > 0.7))
            instance = instance[threshoding]
            inst_class_scores = inst_class_scores[threshoding]
            cls_final = cls_final[threshoding]

            final_dict = {
                "ins": rle_encode_gpu_batch(instance),
                "conf": inst_class_scores.cpu(),
                "final_class": cls_final.cpu(),
            }
            # NOTE Final instance
            torch.save(final_dict, os.path.join(save_dir_final, f"{scene_id}.pth"))

            # NOTE ############################################# VISUALIZING output
            
            print('Visualizing output')
            ply_file = cfg.data.original_ply
            point, color = read_pointcloud(os.path.join(ply_file,scene_id + '.ply'))
            color = color * 127.5
            vis = viz.Visualizer()
            vis.add_points(f'pcl', point, color.astype(np.float32), point_size=20, visible=True)
            save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output) # final_output
            dic = torch.load(os.path.join(save_dir_final, f"{scene_id}.pth"))

            instance = dic['ins']
            instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
            conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)
            label = dic['final_class']

            prompt = class_names + ['others']
            pallete =  generate_palette(int(2e3 + 1))
            tt_col = color.copy()
            limit = 10
            for i in range(0, instance.shape[0]):
                tt_col[instance[i] == 1] = pallete[i]
                if  limit > 0: # be more specific but limit 10 masks (avoiding lag)
                    limit -= 1
                    tt_col_specific = color.copy()
                    tt_col_specific[instance[i] == 1] = pallete[i]
                    vis.add_points(f'final mask: ' + str(i) + '_' + prompt[label[i]], point, tt_col_specific, point_size=20, visible=True)
            vis.add_points(f'final mask: ' + str(i), point, tt_col, point_size=20, visible=True)
            pyviz3d_dir = '../viz' # visualization directory
            vis.save(pyviz3d_dir)
            print('DONE')
