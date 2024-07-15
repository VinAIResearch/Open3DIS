# Reproduce OpenYOLO3D quantitative results (https://github.com/aminebdj/OpenYOLO3D)
# Modified by PhucNDA.

import time
import torch
import os
import os.path as osp
import imageio
import glob
import open3d as o3d
import numpy as np
import math
import colorsys
from tqdm import tqdm, trange
from munch import Munch
import argparse
import yaml
import pycocotools.mask as maskdec
from detectron2.structures import BitMasks
from open3dis.src.mapper import PointCloudToImageMapper

from open3dis.dataset import build_dataset

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


def get_iou(masks):
    masks = masks.float()
    intersection = torch.einsum('ij,kj -> ik', masks, masks)
    num_masks = masks.shape[0]
    masks_batch_size = 2 # scannet 200: 20
    if masks_batch_size < num_masks:
        ratio = num_masks//masks_batch_size
        remaining = num_masks-ratio*masks_batch_size
        start_masks = list(range(0,ratio*masks_batch_size, masks_batch_size))
        if remaining == 0:
            end_masks = list(range(masks_batch_size,(ratio+1)*masks_batch_size,masks_batch_size))
        else:
            end_masks = list(range(masks_batch_size,(ratio+1)*masks_batch_size,masks_batch_size))
            end_masks[-1] = num_masks
    else:
        start_masks = [0]
        end_masks = [num_masks]
    union = torch.cat([((masks[st:ed, None, :]+masks[None, :, :]) >= 1).sum(-1) for st,ed in zip(start_masks, end_masks)])
    iou = torch.div(intersection,union)
    
    return iou

def apply_nms(masks, scores, nms_th):
    masks = masks.permute(1,0)
    scored_sorted, sorted_scores_indices = torch.sort(scores, descending=True)
    inv_sorted_scores_indices = {sorted_id.item(): id for id, sorted_id in enumerate(sorted_scores_indices)}
    maskes_sorted = masks[sorted_scores_indices]
    iou = get_iou(maskes_sorted)
    available_indices = torch.arange(len(scored_sorted))
    for indx in range(len(available_indices)):
        remove_indices = torch.where(iou[indx,indx+1:] > nms_th)[0]
        available_indices[indx+1:][remove_indices] = 0
    remaining = available_indices.unique()
    keep_indices = torch.tensor([inv_sorted_scores_indices[id.item()] for id in remaining])
    return keep_indices


def get_visibility_mat(pred_masks_3d, inside_mask, topk = 15):
    intersection = torch.einsum("ik, fk -> if", pred_masks_3d.float(), inside_mask.float())
    total_point_number = pred_masks_3d[:, None, :].float().sum(dim = -1)
    visibility_matrix = intersection/total_point_number
    
    if topk > visibility_matrix.shape[-1]:
        topk = visibility_matrix.shape[-1]
    
    max_visiblity_in_frame = torch.topk(visibility_matrix, topk, dim = -1).indices
    
    visibility_matrix_bool = torch.zeros_like(visibility_matrix).bool()
    visibility_matrix_bool[torch.tensor(range(len(visibility_matrix_bool)))[:, None],max_visiblity_in_frame] = True
    
    return visibility_matrix_bool

def compute_iou(box, boxes):
    assert box.shape == (4,), "Reference box must be of shape (4,)"
    assert boxes.shape[1] == 4, "Boxes must be of shape (N, 4)"
    
    x1_inter = torch.max(box[0], boxes[:, 0])
    y1_inter = torch.max(box[1], boxes[:, 1])
    x2_inter = torch.min(box[2], boxes[:, 2])
    y2_inter = torch.min(box[3], boxes[:, 3])
    inter_area = (x2_inter - x1_inter).clamp(0) * (y2_inter - y1_inter).clamp(0)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    iou = inter_area / union_area
    
    return iou

class OpenYolo3D():
    def __init__(self, scene_id, cfg):
        '''
        We already have precomputed 3D masks and 2D masks (high-granularity)
        '''

        self.cfg = cfg
        self.scene_id = scene_id
        
        # Make dir
        save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output) # final_output
        os.makedirs(save_dir_final, exist_ok=True)

        # 3D proposals
        data_path = './data/Scannet200/Scannet200_3D/Scannet200_open3dis_ISBNet_DETIC'
        agnostic3d_path = os.path.join(data_path, f"{scene_id}.pth")
        agnostic3d_data = torch.load(agnostic3d_path)
        instance_3d_encoded = np.array(agnostic3d_data["ins"])
        n_instance_3d = instance_3d_encoded.shape[0]
        if isinstance(instance_3d_encoded[0], dict):
            instance_3d = torch.stack([torch.from_numpy(rle_decode(in3d)) for in3d in instance_3d_encoded], dim=0)
        else:
            instance_3d = torch.stack([torch.tensor(in3d) for in3d in instance_3d_encoded], dim=0)
        # 2D-lifted-3D proposals -> notyet        
        self.instance = torch.cat([instance_3d], dim=0)
        
        # 2D proposals
        self.preds_2d = torch.load(os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.mask2d_output, scene_id+'.pth'))

        # 2D data loader
        scene_dir = os.path.join(cfg.data.datapath, scene_id)
        self.loader = build_dataset(root_path=scene_dir, cfg=cfg)
        self.points = self.loader.read_pointcloud()
        self.points = torch.from_numpy(self.points).cuda()
        # Pointcloud Image mapper
        img_dim = cfg.data.img_dim
        self.pointcloud_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=self.loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary)
        self.label_maps, self.score_maps = self.construct_label_maps()
        final_classes, confi = self.getmasklabel()
        # FreeVocab results
        masks = []
        classes = []
        conf = []
        for i in range(len(final_classes)):
            masks.append(self.instance[i])
            classes.append(final_classes[i])
            conf.append(confi[i])
        torch.save({'masks': masks, 'class' : classes, 'confidence_3d': conf}, os.path.join(save_dir_final, f"{scene_id}.pth"))

        
    def construct_label_maps(self):
        class_names = []
        frame_num = 0
        self.mapping = []
        for i in trange(0, len(self.loader), cfg.data.img_interval):
            frame = self.loader[i]
            frame_id = frame["frame_id"]
            depth = frame['depth_path']  # (h, w)
            pose = frame['pose_path']  # (4,4) - camera pose
            pose = self.loader.read_pose(pose)
            depth = self.loader.read_depth(depth)
            #### Point mapping ####
            n_points = self.points.shape[0]
            mapping = torch.ones([n_points, 4], dtype=int, device=self.points.device)
            mapping[:, 1:4] = self.pointcloud_mapper.compute_mapping_torch(pose, self.points, depth, intrinsic = frame["scannet_depth_intrinsic"])
            #######################
            self.mapping.append(mapping.cpu())
            frame_num += 1

            # empty masks
            if frame_id not in self.preds_2d.keys():
                continue

            for name in self.preds_2d[frame_id]['class']:
                class_names.append(name)
        self.class_names = list(set(class_names))

        # DETIC, ScanNet200 label map (480,640)
        score_maps = (torch.zeros((frame_num, self.cfg.data.img_dim[1], self.cfg.data.img_dim[0]))*-1).type(torch.float32)
        label_maps = (torch.ones((frame_num, self.cfg.data.img_dim[1], self.cfg.data.img_dim[0]))*-1).type(torch.int32)
        for i in trange(0, len(self.loader), cfg.data.img_interval):
            frame = self.loader[i]
            frame_id = frame["frame_id"]

            # empty masks
            if frame_id not in self.preds_2d.keys():
                continue

            labels = self.preds_2d[frame_id]['class']
            conf = torch.tensor(self.preds_2d[frame_id]['conf'])
            encoded_masks = self.preds_2d[frame_id]['masks']
            BOXES = []
            if encoded_masks is not None:
                masks = []
                for mask in encoded_masks:
                    masks.append(torch.tensor(maskdec.decode(mask)))
                masks = torch.stack(masks, dim=0).cpu() # cuda fast but OOM
                pred = BitMasks(masks)
                BOXES = pred.get_bounding_boxes()
            boxes = []
            for box in BOXES:
                boxes.append(box)
            boxes = torch.stack(boxes).long()
            bboxes = boxes.long()
            labels_id = []
            for name in labels:
                labels_id.append(self.class_names.index(name))
            labels_id = torch.tensor(labels_id, dtype = torch.int32)

            # bboxes[:,0] = bboxes[:,0]*self.scaling_params[1]
            # bboxes[:,2] = bboxes[:,2]*self.scaling_params[1]
            # bboxes[:,1] = bboxes[:,1]*self.scaling_params[0]
            # bboxes[:,3] = bboxes[:,3]*self.scaling_params[0]
            bboxes_weights = (bboxes[:,2]-bboxes[:,0])+(bboxes[:,3]-bboxes[:,1])
            sorted_indices = bboxes_weights.sort(descending=True).indices
            bboxes = bboxes[sorted_indices]
            labels_id = labels_id[sorted_indices]
            conf = conf[sorted_indices]

            for id, bbox in enumerate(bboxes):
                score_maps[i//cfg.data.img_interval, bbox[1]:bbox[3],bbox[0]:bbox[2]] = conf[id]
                label_maps[i//cfg.data.img_interval, bbox[1]:bbox[3],bbox[0]:bbox[2]] = labels_id[id]

        return label_maps, score_maps
    
    def getmasklabel(self):
        final_classes = []
        confidence = []
        for mask in tqdm(self.instance):
            ious = []
            # boxes= []
            for i in range(len(self.mapping)):
                mapping = self.mapping[i]
                intersect = torch.logical_and(mask,mapping[:,3]).sum()
                union = torch.logical_or(mask,mapping[:,3]).sum()
                iou = intersect / union
                ious.append(iou)
                # projection = torch.logical_and(mask, mapping[:,3])
                # if projection.sum().item() == 0:
                #     boxes.append([0,0,0,0])
                # else:
                #     xt,yt = mapping[projection,1].min().item(), mapping[projection,2].min().item()
                #     xb,yb = mapping[projection,1].max().item(), mapping[projection,2].max().item()
                #     boxes.append([xt,yt,xb,yb])
            # boxes = torch.tensor(boxes)
            ious = torch.stack(ious)
            values, indices = torch.topk(ious, k=min(len(ious),40))
            # Selected labels
            distribution = []
            scotribution = []
            for id in indices:
                mapping = self.mapping[id]
                projection = torch.logical_and(mask, mapping[:,3])
                if projection.sum()==0:
                    break
                xt,yt = mapping[projection,1].min().item(), mapping[projection,2].min().item()
                xb,yb = mapping[projection,1].max().item(), mapping[projection,2].max().item()
                distribution.append(self.label_maps[id, xt:xb, yt:yb].clone().view(-1))
                scotribution.append(self.score_maps[id, xt:xb, yt:yb].clone().view(-1))

            if len(distribution) == 0:
                final_classes.append('None')
                confidence.append(0.0)
                continue
            final_class, _ = torch.cat(distribution).mode()
            ### err
            temp = (torch.cat(distribution) == final_class.item())
            confidence.append(torch.cat(scotribution)[temp].mean().item())
            final_classes.append(self.class_names[final_class.item()])
        return final_classes, confidence


def get_parser():
    parser = argparse.ArgumentParser(description="Data Configuration Like Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    # Scannet split path
    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])
    for scene_id in tqdm(scene_ids):
        ## Tracker
        done = False
        path = scene_id + ".pth"
        print('Process: ', path)
        with open("tracker_lifted.txt", "r") as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
            for line in lines:
                if path in line:
                    done = True
                    break
        if done == True:
            print("existed " + path)
            continue
        ## Write append each line
        with open("tracker_lifted.txt", "a") as file:
            file.write(path + "\n")

        model = OpenYolo3D(scene_id, cfg)