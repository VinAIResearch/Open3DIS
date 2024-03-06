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

# Grounding DINO
from detectron2.structures import BitMasks
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from munch import Munch
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from open3dis.dataset.scannet_loader import ScanNetReader, scaling_mapping

#### Open3DIS util
from open3dis.src.fusion_util import NMS_cuda
from open3dis.src.mapper import PointCloudToImageMapper
from PIL import Image, ImageDraw, ImageFont

# SAM
from segment_anything import SamPredictor, build_sam, build_sam_hq
from tqdm import tqdm, trange


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
    scannet_loader = ScanNetReader(root_path=scene_dir, cfg=cfg)

    # Pointcloud Image mapper
    img_dim = cfg.data.img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=scannet_loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
    )

    points = scannet_loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    n_points = points.shape[0]

    grounded_data_dict = {}

    # Accmulate CLIP mask feature onto 3D point cloud ?
    if gen_feat:
        grounded_features = torch.zeros((n_points, cfg.foundation_model.clip_dim)).cuda()
    else:
        grounded_features = None

    for i in trange(0, len(scannet_loader), cfg.data.img_interval):
        frame = scannet_loader[i]
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
            image = scannet_loader.load_image(image_path)
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            plt.axis("off")
            # plot out
            plt.savefig(
                os.path.join("../debug/" + scene_id + "/sam_" + str(i) + ".jpg"),
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
            pose = scannet_loader.read_pose(frame["pose_path"])
            depth = scannet_loader.read_depth(frame["depth_path"])
            mapping = torch.ones([n_points, 4], dtype=int, device="cuda")
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)
            if "scannet" in cfg.data.dataset_name:  # Scaling mapper in Scannet only
                new_mapping = scaling_mapping(
                    torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[0], rgb_img_dim[1]
                )
                mapping[:, 1:4] = torch.cat((new_mapping, mapping[:, 3].unsqueeze(1)), dim=1)

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


if __name__ == "__main__":

    # Multiprocess logger
    if os.path.exists("tracker_2d.txt") == False:
        with open("tracker_2d.txt", "w") as file:
            file.write("Processed Scenes .\n")
    cfg = Munch.fromDict(yaml.safe_load(open("./configs/scannet200.yaml", "r").read()))

    # Fondation model loader
    clip_adapter, clip_preprocess, grounding_dino_model, sam_predictor = init_foundation_models(cfg)

    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

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

            print("Process", scene_id)
            grounded_data_dict, grounded_features = gen_grounded_mask_and_feat(
                scene_id,
                clip_adapter,
                clip_preprocess,
                grounding_dino_model,
                sam_predictor,
                class_names=INSTANCE_CAT_SCANNET_200,
                cfg=cfg,
            )

            # Save PC features
            torch.save({"feat": grounded_features.half()}, os.path.join(save_dir_feat, scene_id + ".pth"))
            # Save 2D mask
            torch.save(grounded_data_dict, os.path.join(save_dir, scene_id + ".pth"))

            torch.cuda.empty_cache()
