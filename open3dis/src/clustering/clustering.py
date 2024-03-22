import itertools
import math
import os
import random
import time

import cv2
import numpy as np
import open3d as o3d
import pycocotools
import pyviz3d.visualizer as viz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from open3dis.dataset.scannet_loader import ScanNetReader, scaling_mapping
from open3dis.src.clustering.clustering_utils import (
    compute_projected_pts,
    compute_projected_pts_torch,
    compute_relation_matrix_self,
    compute_visibility_mask,
    compute_visibility_mask_torch,
    compute_visible_masked_pts,
    compute_visible_masked_pts_torch,
    custom_scatter_mean,
    find_connected_components,
    read_detectron_instances,
    resolve_overlapping_masks,
)
from open3dis.src.fusion_util import NMS_cuda
from open3dis.src.mapper import PointCloudToImageMapper
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm import tqdm, trange


# Hierachical merge
def hierarchical_agglomerative_clustering(
    pcd_list,
    left,
    right,
    spp,
    n_spp,
    n_points,
    sieve,
    detic=False,
    visi=0.9,
    simi=0.5,
):
    global num_point, dc_feature_matrix, dc_feature_spp
    if left == right:
        device = spp.device
        # Graph initialization
        index = left
        masks = pcd_list[index]["masks"]
        mapping = pcd_list[index]["mapping"].cuda()

        total_spp_points = torch_scatter.scatter((mapping[:, 3] == 1).float(), spp, dim=0, reduce="sum")

        weights = []
        ### Per mask processing
        mask3d = []

        if masks == None:
            return [], []

        for m, mask in enumerate(masks):
            spp_weights = torch.zeros((n_spp), dtype=torch.float32, device=device)
            idx = torch.nonzero(mapping[:, 3] == 1).view(-1)
            highlight_points = idx[
                mask[mapping[idx][:, [1, 2]][:, 0], mapping[idx][:, [1, 2]][:, 1]].nonzero(as_tuple=True)[0]
            ].long()

            sieve_mask = torch.zeros((n_points), device=device)
            sieve_mask[highlight_points] = 1

            num_related_points = torch_scatter.scatter(sieve_mask.float(), spp, dim=0, reduce="sum")

            spp_weights = torch.where(
                total_spp_points == 0, torch.tensor(0.0).cuda(), num_related_points / total_spp_points
            )
            target_spp = torch.nonzero(spp_weights >= 0.5).view(-1)

            if len(target_spp) <= 1:
                continue

            elif len(target_spp) == 1:

                target_weight = torch.zeros_like(spp_weights)
                target_weight[target_spp] = spp_weights[target_spp]

                group_tmp = torch.zeros((n_spp), dtype=torch.int, device=device)
                group_tmp[target_spp] = 1

                mask3d.append(group_tmp)
                weights.append(spp_weights)

            else:
                pairwise_dc_dist = dc_feature_matrix[target_spp, :][:, target_spp]
                pairwise_dc_dist[torch.eye((len(target_spp)), dtype=torch.bool, device=dc_feature_matrix.device)] = -10
                max_dc_dist = torch.max(pairwise_dc_dist, dim=1)[0]

                valid_spp = max_dc_dist >= 0.5

                if valid_spp.sum() > 0:
                    target_spp = target_spp[valid_spp]

                    target_weight = torch.zeros_like(spp_weights)
                    target_weight[target_spp] = spp_weights[target_spp]

                    group_tmp = torch.zeros((n_spp), dtype=torch.int, device=device)
                    group_tmp[target_spp] = 1

                    mask3d.append(group_tmp)
                    weights.append(spp_weights)

        if len(mask3d) == 0:
            return [], []
        mask3d = torch.stack(mask3d, dim=0)
        weights = torch.stack(weights, dim=0)
        return mask3d, weights

    mid = int((left + right) / 2)
    graph_1_onehot, weight_1 = hierarchical_agglomerative_clustering(
        pcd_list, left, mid, spp, n_spp, n_points, sieve, detic=detic
    )
    graph_2_onehot, weight_2 = hierarchical_agglomerative_clustering(
        pcd_list, mid + 1, right, spp, n_spp, n_points, sieve, detic=detic
    )

    if len(graph_1_onehot) == 0 and len(graph_2_onehot) == 0:
        return [], []

    if len(graph_1_onehot) == 0:
        return graph_2_onehot, weight_2

    if len(graph_2_onehot) == 0:
        return graph_1_onehot, weight_1

    new_graph = torch.cat([graph_1_onehot, graph_2_onehot], dim=0)
    new_weight = torch.cat([weight_1, weight_2], dim=0)

    graph_feat = new_graph.bool().float() @ dc_feature_spp  # n, f

    graph_feat_matrix = pairwise_cosine_similarity(graph_feat, graph_feat)

    iou_matrix, _, recall_matrix = compute_relation_matrix_self(new_graph, spp, sieve)
    adjacency_matrix = ((iou_matrix >= 0.9) | (recall_matrix >= 0.9)) & (graph_feat_matrix >= 0.9)
    # adjacency_matrix = ((iou_matrix >= 0.8) | (recall_matrix >= 0.8)) & (graph_feat_matrix >= 0.8)
    # adjacency_matrix = ((iou_matrix >= visi) | (recall_matrix >= visi)) & (graph_feat_matrix >= 0.9)
    # adjacency_matrix = ((iou_matrix >= visi) | (recall_matrix >= visi)) & (graph_feat_matrix >= simi)
    # adjacency_matrix = ((iou_matrix >= 0.9) | (recall_matrix >= 0.9))
    adjacency_matrix = adjacency_matrix | adjacency_matrix.T

    # if adjacency_matrix
    if adjacency_matrix.sum() == new_graph.shape[0]:
        return new_graph, new_weight

    # merge instances based on the adjacency matrix
    connected_components = find_connected_components(adjacency_matrix)
    M = len(connected_components)

    merged_instance = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.int, device=graph_2_onehot.device)
    merged_weight = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.float, device=graph_2_onehot.device)

    for i, cluster in enumerate(connected_components):
        merged_instance[i] = new_graph[cluster].sum(0)
        merged_weight[i] = new_weight[cluster].mean(0)

    new_graph = merged_instance
    new_weight = merged_weight

    return new_graph, new_weight


# Sequential merge
def sequential_feature_overlap_feat_agg(
    pcd_list,
    left,
    right,
    spp,
    n_spp,
    n_points,
    detic=False,
    visi=0.5,
    simi=0.5,):
    global distance_matrix, num_instance, num_point, pointfeature, dc_feature_matrix, dc_feature_spp

    def single(pcd_list, left, right, spp, n_spp, n_points, detic=False, visi=0.5, simi=0.5):
        device = spp.device
        # Graph initialization
        index = left
        masks = pcd_list[index]["masks"]
        mapping = pcd_list[index]["mapping"]

        total_spp_points = torch_scatter.scatter((mapping[:, 3] == 1).float(), spp, dim=0, reduce="sum")

        weights = []
        ### Per mask processing
        mask3d = []
        if masks == None:
            return [], []
        for m, mask in enumerate(masks):
            spp_weights = torch.zeros((n_spp), dtype=torch.float32, device=device)
            idx = torch.nonzero(mapping[:, 3] == 1).view(-1)
            highlight_points = idx[
                mask[mapping[idx][:, [1, 2]][:, 0], mapping[idx][:, [1, 2]][:, 1]].nonzero(as_tuple=True)[0]
            ].long()

            sieve_mask = torch.zeros((n_points), device=device)
            sieve_mask[highlight_points] = 1

            num_related_points = torch_scatter.scatter(sieve_mask.float(), spp, dim=0, reduce="sum")

            spp_weights = torch.where(total_spp_points == 0, 0, num_related_points / total_spp_points)

            target_spp = torch.nonzero(spp_weights >= visi).view(-1)

            if len(target_spp) <= 1:
                continue

            elif len(target_spp) == 1:

                target_weight = torch.zeros_like(spp_weights)
                target_weight[target_spp] = spp_weights[target_spp]

                group_tmp = torch.zeros((n_spp), dtype=torch.int, device=device)
                group_tmp[target_spp] = 1

                mask3d.append(group_tmp)
                weights.append(spp_weights)

            else:
                pairwise_dc_dist = dc_feature_matrix[target_spp, :][:, target_spp]
                pairwise_dc_dist[torch.eye((len(target_spp)), dtype=torch.bool, device=dc_feature_matrix.device)] = -10
                max_dc_dist = torch.max(pairwise_dc_dist, dim=1)[0]

                valid_spp = max_dc_dist >= 0.5

                if valid_spp.sum() > 0:
                    target_spp = target_spp[valid_spp]

                    target_weight = torch.zeros_like(spp_weights)
                    target_weight[target_spp] = spp_weights[target_spp]

                    group_tmp = torch.zeros((n_spp), dtype=torch.int, device=device)
                    group_tmp[target_spp] = 1

                    mask3d.append(group_tmp)
                    weights.append(spp_weights)

        if len(mask3d) == 0:
            return [], []
        mask3d = torch.stack(mask3d, dim=0)
        weights = torch.stack(weights, dim=0)
        return mask3d, weights

    graph_1_onehot, weight_1 = [], []
    for i in range(right):
        graph_2_onehot, weight_2 = single(pcd_list, i, right, spp, n_spp, n_points, detic=detic)

        if len(graph_1_onehot) == 0 and len(graph_2_onehot) == 0:
            continue

        if len(graph_1_onehot) == 0:
            graph_1_onehot, weight_1 = graph_2_onehot, weight_2

        if len(graph_2_onehot) == 0:
            continue

        new_graph = torch.cat([graph_1_onehot, graph_2_onehot], dim=0)
        new_weight = torch.cat([weight_1, weight_2], dim=0)

        # while True:
        graph_feat = new_graph.bool().float() @ dc_feature_spp  # n, f

        graph_feat_matrix = pairwise_cosine_similarity(graph_feat, graph_feat)

        iou_matrix, _, recall_matrix = compute_relation_matrix_self(new_graph[:, spp])
        adjacency_matrix = ((iou_matrix >= visi) | (recall_matrix >= visi)) & (graph_feat_matrix >= simi)
        adjacency_matrix = adjacency_matrix | adjacency_matrix.T

        # if adjacency_matrix
        if adjacency_matrix.sum() == new_graph.shape[0]:
            continue

        # merge instances based on the adjacency matrix
        connected_components = find_connected_components(adjacency_matrix)
        M = len(connected_components)

        merged_instance = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.int, device=graph_2_onehot.device)
        merged_weight = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.float, device=graph_2_onehot.device)

        for i, cluster in enumerate(connected_components):
            merged_instance[i] = new_graph[cluster].sum(0)
            merged_weight[i] = new_weight[cluster].mean(0)

        graph_1_onehot = merged_instance
        weight_1 = merged_weight

    return graph_1_onehot, weight_1


distance_matrix = None
distance_matrix2 = None
num_instance = None
num_point = None
pointfeature = None
point_feat = None
dc_feature_matrix = None
dc_feature_spp = None
means_spp = None


def process_hierarchical_agglomerative(scene_id, cfg):
    global num_instance, num_point, dc_feature_matrix, dc_feature_spp

    visi = cfg.cluster.visi
    simi = cfg.cluster.simi

    exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)

    spp_path = os.path.join(cfg.data.spp_path, f"{scene_id}.pth")
    
    mask2d_path = os.path.join(exp_path, cfg.exp.mask2d_output, scene_id + ".pth")
    # mask2d_path = os.path.join(exp_path, "maskGdino", scene_id + ".pth")
    # mask2d_path = os.path.join(exp_path, 'maskDetic', scene_id+'.pth')
    # mask2d_path = os.path.join(exp_path, 'maskODISE', scene_id+'.pth')
    # mask2d_path = os.path.join(exp_path, 'maskSEEM', scene_id+'.pth')

    dc_feature_path = os.path.join(cfg.data.dc_features_path, scene_id + ".pth")

    scene_dir = os.path.join(cfg.data.datapath, scene_id)
    scannet_loader = ScanNetReader(root_path=scene_dir, cfg=cfg)

    img_dim = cfg.data.img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=scannet_loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
    )

    points = scannet_loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    n_points = points.shape[0]

    spp = torch.tensor(torch.load(spp_path)).cuda() # memory ease
    n_spp = torch.unique(spp).shape[0]
    unique_spp, spp, num_point = torch.unique(spp, return_inverse=True, return_counts=True)

    dc_feature = torch.load(dc_feature_path).cuda().float()

    dc_feature_spp = torch_scatter.scatter(dc_feature, spp, dim=0, reduce="sum")
    dc_feature_spp = F.normalize(dc_feature_spp, dim=1, p=2)
    dc_feature_matrix = pairwise_cosine_similarity(dc_feature_spp, dc_feature_spp)

    visibility = torch.zeros((n_points), dtype=torch.int, device=spp.device)

    sieve = [] # number of point in spp for fast calculating IoU between 3D masks
    for i in range (n_spp):
        sieve.append((spp == i).sum().item()) 
    sieve = torch.tensor(sieve)

    groundedsam_data_dict = torch.load(mask2d_path)
    pcd_list = []

    for i in trange(0, len(scannet_loader), cfg.data.img_interval):
        frame = scannet_loader[i]
        frame_id = frame["frame_id"]

        if frame_id not in groundedsam_data_dict.keys():
            continue

        groundedsam_data = groundedsam_data_dict[frame_id]

        pose = scannet_loader.read_pose(frame["pose_path"])
        depth = scannet_loader.read_depth(frame["depth_path"])
        rgb_img = scannet_loader.read_image(frame["image_path"])

        rgb_img_dim = rgb_img.shape[:2]

        encoded_masks = groundedsam_data["masks"]

        masks = None
        if encoded_masks is not None:
            masks = []
            for mask in encoded_masks:
                masks.append(torch.tensor(pycocotools.mask.decode(mask)))
            masks = torch.stack(masks, dim=0).cpu() # cuda fast but OOM

        if "scannetpp" in cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
            depth = cv2.resize(depth, (img_dim[0], img_dim[1]))
            mapping = torch.ones([n_points, 4], dtype=int, device="cuda")
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)

        if "scannet200" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)
            new_mapping = scaling_mapping(
                torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[0], rgb_img_dim[1]
            )
            mapping[:, 1:4] = torch.cat((new_mapping, mapping[:, 3].unsqueeze(1)), dim=1)

        visibility[mapping[:, 3] == 1] += 1

        dic = {"mapping": mapping.cpu(), "masks": masks}
        pcd_list.append(dic)
    
    torch.cuda.empty_cache()
    num_instance = 0
    
    groups, weights = hierarchical_agglomerative_clustering(
        pcd_list, 0, len(pcd_list) - 1, spp, n_spp, n_points, sieve, detic=False, visi=visi, simi=simi
    )

    if len(groups) == 0:
        return None, None

    proposals_pred = groups[:, spp]  # .bool()

    inst_visibility = proposals_pred / visibility.clip(min=1e-6)[None, :]
    proposals_pred[inst_visibility < cfg.cluster.point_visi] = 0

    confidence = (groups.bool() * weights).sum(dim=1) / groups.sum(dim=1)

    del groups, weights

    proposals_pred = proposals_pred.bool()

    proposals_pred_final = custom_scatter_mean(
        proposals_pred,
        spp[None, :].expand(len(proposals_pred), -1),
        dim=-1,
        pool=True,
        output_type=torch.float32,
    )
    proposals_pred = (proposals_pred_final >= 0.5)[:, spp]

    mask_valid = proposals_pred.sum(1) > cfg.cluster.valid_points
    proposals_pred = proposals_pred[mask_valid].cpu()
    confidence = confidence[mask_valid].cpu()

    return proposals_pred, confidence
