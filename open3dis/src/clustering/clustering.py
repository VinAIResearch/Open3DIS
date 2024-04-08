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
from open3dis.dataset import build_dataset
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
import matplotlib.pyplot as plt


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
    visi=0.7,
    reca = 1.0,
    simi=0.5,
    iterative=True,
):
    global num_point, dc_feature_matrix, dc_feature_spp
    if left == right:
        device = spp.device
        # Graph initialization
        index = left

        if pcd_list[index]["masks"] is None:
            return [], []
        
        masks = pcd_list[index]["masks"].cuda()
        mapping = pcd_list[index]["mapping"].cuda()

        total_spp_points = torch_scatter.scatter((mapping[:, 3] == 1).float(), spp, dim=0, reduce="sum")

        weights = []
        ### Per mask processing
        mask3d = []

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
                total_spp_points==0, 0, num_related_points / total_spp_points
            )
            target_spp = torch.nonzero(spp_weights >= 0.5).view(-1)

            if len(target_spp) <= 1:
                continue

            elif len(target_spp) == 1:

                target_weight = torch.zeros_like(spp_weights)
                target_weight[target_spp] = spp_weights[target_spp]

                group_tmp = torch.zeros((n_spp), dtype=torch.int8, device=device)
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

                    group_tmp = torch.zeros((n_spp), dtype=torch.int8, device=device)
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
        pcd_list, left, mid, spp, n_spp, n_points, sieve, detic=detic, visi = visi, reca = reca, simi = simi, iterative=iterative
    )
    graph_2_onehot, weight_2 = hierarchical_agglomerative_clustering(
        pcd_list, mid + 1, right, spp, n_spp, n_points, sieve, detic=detic, visi = visi, reca = reca, simi = simi, iterative=iterative
    )

    if len(graph_1_onehot) == 0 and len(graph_2_onehot) == 0:
        return [], []

    if len(graph_1_onehot) == 0:
        return graph_2_onehot, weight_2

    if len(graph_2_onehot) == 0:
        return graph_1_onehot, weight_1

    if iterative:
        new_graph = torch.cat([graph_1_onehot, graph_2_onehot], dim=0)
        new_weight = torch.cat([weight_1, weight_2], dim=0)

        graph_feat = new_graph.bool().float() @ dc_feature_spp  # n, f

        graph_feat_matrix = pairwise_cosine_similarity(graph_feat, graph_feat)

        iou_matrix, _, recall_matrix = compute_relation_matrix_self(new_graph, spp, sieve)
        # iou_matrix, _, recall_matrix = compute_relation_matrix_self(new_graph)
        
        #####
        adjacency_matrix = (iou_matrix >= visi)
        if reca < 0.98:
            adjacency_matrix |= (recall_matrix >= reca)    
        if simi > 0.1: # scannetpp using 3D features from 3D backbone pretrained scannet200 yeilds not good results
            adjacency_matrix &= (graph_feat_matrix >= simi)
        adjacency_matrix = adjacency_matrix | adjacency_matrix.T
        #####

        # if adjacency_matrix
        if adjacency_matrix.sum() == new_graph.shape[0]:
            return new_graph, new_weight

        # merge instances based on the adjacency matrix
        connected_components = find_connected_components(adjacency_matrix)
        M = len(connected_components)

        merged_instance = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.int8, device=graph_2_onehot.device)
        merged_weight = torch.zeros((M, graph_2_onehot.shape[1]), dtype=torch.float, device=graph_2_onehot.device)

        for i, cluster in enumerate(connected_components):
            merged_instance[i] = new_graph[cluster].sum(0)
            merged_weight[i] = new_weight[cluster].mean(0)

        new_graph = merged_instance
        new_weight = merged_weight

        return new_graph, new_weight
    
    new_graph, new_weight = [], [] 

    vis1 = torch.zeros((graph_1_onehot.shape[0]), device=graph_2_onehot.device)
    vis2 = torch.zeros((graph_2_onehot.shape[0]), device=graph_2_onehot.device)

    intersections = graph_1_onehot[:, spp].float() @ graph_2_onehot[:, spp].float().T
    # ious = intersections / ()
    # intersections = ((torch.logical_and(graph_1_onehot.bool()[:, None, :], graph_2_onehot.bool()[None, :, :])) * num_point[None, None]).sum(dim=-1)
    ious = intersections / ((graph_1_onehot.long() * num_point).sum(1)[:, None] + (graph_2_onehot.long() * num_point).sum(1)[None, :] - intersections)
    
    # similar_matrix = F.cosine_similarity(graph_1_feat[:, None, :], graph_2_feat[None, :, :], dim=2)
    graph_1_feat = torch.einsum('pn,nc->pc', graph_1_onehot.float(), dc_feature_spp) #/ torch.sum(graph_1_onehot, dim=1, keepdim=True)
    graph_2_feat = torch.einsum('pn,nc->pc', graph_2_onehot.float(), dc_feature_spp) #/ torch.sum(graph_2_onehot, dim=1, keepdim=True)
    similar_matrix = pairwise_cosine_similarity(graph_1_feat, graph_2_feat)
    
    row_inds = torch.arange((ious.shape[0]), dtype=torch.long, device=graph_1_onehot.device)
    max_ious, col_inds = torch.max(ious, dim=-1)
    valid_mask = (max_ious > visi) & (similar_matrix[row_inds, col_inds] > simi)
    
    row_inds_ = row_inds[valid_mask]
    col_inds_ = col_inds[valid_mask]
    vis2[col_inds_] = 1
    vis1[row_inds_] = 1

    union_masks = (graph_1_onehot[row_inds_] + graph_2_onehot[col_inds_]).int()
    intersection_masks = (graph_1_onehot[row_inds_] * graph_2_onehot[col_inds_]).bool()

    union_weight = 0.5 * (weight_1[row_inds_] + weight_2[col_inds_]) * intersection_masks \
                 + weight_1[row_inds_] * graph_1_onehot[row_inds_] \
                 + weight_2[col_inds_] * graph_2_onehot[col_inds_] 
    
    temp = (intersection_masks.float() + graph_1_onehot[row_inds_].float() + graph_2_onehot[col_inds_].float())
    union_weight = torch.where(temp == 0, 0, union_weight / temp)

    new_graph.append(union_masks.bool())
    new_weight.append(union_weight)

    nomatch_inds_group1 = torch.nonzero(vis1 == 0).view(-1)
    new_graph.append(graph_1_onehot[nomatch_inds_group1])
    new_weight.append(weight_1[nomatch_inds_group1])

    nomatch_inds_group2 = torch.nonzero(vis2 == 0).view(-1)
    new_graph.append(graph_2_onehot[nomatch_inds_group2])
    new_weight.append(weight_2[nomatch_inds_group2])


    new_graph = torch.cat(new_graph, dim=0)
    new_weight = torch.cat(new_weight, dim=0)

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
    reca = cfg.cluster.recall
    iterative = cfg.cluster.iterative if hasattr(cfg.cluster, 'iterative') else True

    exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)

    spp_path = os.path.join(cfg.data.spp_path, f"{scene_id}.pth")
    
    mask2d_path = os.path.join(exp_path, cfg.exp.mask2d_output, scene_id + ".pth")
    # mask2d_path = os.path.join(exp_path, "maskGdino", scene_id + ".pth")
    # mask2d_path = os.path.join(exp_path, 'maskDetic', scene_id+'.pth')
    # mask2d_path = os.path.join(exp_path, 'maskODISE', scene_id+'.pth')
    # mask2d_path = os.path.join(exp_path, 'maskSEEM', scene_id+'.pth')

    dc_feature_path = os.path.join(cfg.data.dc_features_path, scene_id + ".pth")

    scene_dir = os.path.join(cfg.data.datapath, scene_id)
    loader = build_dataset(root_path=scene_dir, cfg=cfg)

    img_dim = cfg.data.img_dim
    pointcloud_mapper = PointCloudToImageMapper(
        image_dim=img_dim, intrinsics=loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
    )

    points = loader.read_pointcloud()
    points = torch.from_numpy(points).cuda()
    n_points = points.shape[0]

    # spp = torch.tensor(torch.load(spp_path)).cuda() # memory ease
    # n_spp = torch.unique(spp).shape[0]
    spp = loader.read_spp(spp_path)
    unique_spp, spp, num_point = torch.unique(spp, return_inverse=True, return_counts=True)
    n_spp = len(unique_spp)

    dc_feature = loader.read_feature(dc_feature_path)

    dc_feature_spp = torch_scatter.scatter(dc_feature, spp, dim=0, reduce="sum")
    dc_feature_spp = F.normalize(dc_feature_spp, dim=1, p=2)
    dc_feature_matrix = pairwise_cosine_similarity(dc_feature_spp, dc_feature_spp)

    visibility = torch.zeros((n_points), dtype=torch.int, device=spp.device)

    sieve = [] # number of point in spp for fast calculating IoU between 3D masks
    for i in range (n_spp):
        sieve.append((spp == i).sum().item()) 
    sieve = torch.tensor(sieve)

    # FIXME 

    # breakpoint()
    # mask2d_path = f'/home/tdngo/Workspace/3dis_ws/Open3DInstanceSegmentation/Dataset/replica/version_final/maskGdino0404conf/{scene_id}.pth'
    groundedsam_data_dict = torch.load(mask2d_path)
    pcd_list = []

    for i in trange(0, len(loader), cfg.data.img_interval):
        frame = loader[i]
        frame_id = frame["frame_id"]
        
        # FIXME
        if frame_id not in groundedsam_data_dict.keys():
            if cfg.data.dataset_name == 'scannetpp':
                if 'frame_'+str(int(frame_id) * 10) in groundedsam_data_dict.keys(): # conflict resolve generating scannet++ issues
                    frame_id = 'frame_'+str(int(frame_id) * 10)
                else:
                    print('skip: ', frame_id)
            else:
                print('skip: ', frame_id)
                continue


        groundedsam_data = groundedsam_data_dict[frame_id]
        encoded_masks = groundedsam_data["masks"]

        pose = loader.read_pose(frame["pose_path"])
        depth = loader.read_depth(frame["depth_path"])
        rgb_img = loader.read_image(frame["image_path"])

        rgb_img_dim = rgb_img.shape[:2]
        

        masks = None
        if encoded_masks is not None:
            masks = []
            for mask in encoded_masks:
                masks.append(torch.tensor(pycocotools.mask.decode(mask)))
            masks = torch.stack(masks, dim=0).cpu() # cuda fast but OOM

        if False:
            # draw output image
            image = rgb_img
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks[:5]:
                show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
            plt.axis("off")
            # plot out
            try:
                os.makedirs("../debug/" + scene_id)
            except:
                pass
            plt.savefig(
                os.path.join("../debug/" + scene_id + "/sam_" + str(i) + ".jpg"),
                bbox_inches="tight",
                dpi=300,
                pad_inches=0.0,
            )

        # breakpoint()
        if "scannetpp" in cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
            depth = cv2.resize(depth, (img_dim[0], img_dim[1]))
            mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic = frame["translated_intrinsics"])

        elif "scannet200" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth) # global intrinsic
            new_mapping = scaling_mapping(
                torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[0], rgb_img_dim[1]
            )
            mapping[:, 1:4] = torch.cat((new_mapping, mapping[:, 3].unsqueeze(1)), dim=1)
        
        elif "replica" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)
            # new_mapping = pointcloud_mapper(torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[0], rgb_img_dim[1])
            # mapping[:, 1:4] = torch.cat((new_mapping,mapping[:,3].unsqueeze(1)),dim=1)
        elif "s3dis" in cfg.data.dataset_name:
            mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
            mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic=frame["intrinsics"])
        else:
            raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

        visibility[mapping[:, 3] == 1] += 1

        dic = {"mapping": mapping.cpu(), "masks": masks}
        pcd_list.append(dic)
    
    torch.cuda.empty_cache()
    num_instance = 0
    
    # breakpoint()
    groups, weights = hierarchical_agglomerative_clustering(pcd_list, 0, len(pcd_list) - 1, spp, n_spp, n_points, sieve, detic=False, visi=visi, reca = reca, simi=simi, iterative=iterative)

    if len(groups) == 0:
        return None, None

    confidence = (groups.bool() * weights).sum(dim=1) / groups.sum(dim=1)
    groups = groups.to(torch.int64).cpu()

    spp = spp.cpu()
    proposals_pred = groups[:, spp]  # .bool()
    del groups, weights
    torch.cuda.empty_cache()

    ## These lines take a lot of memory # achieveing in paper result-> unlock this
    if cfg.cluster.point_visi > 0:
        start = 0
        end = proposals_pred.shape[0]
        inst_visibility = torch.zeros_like(proposals_pred, dtype=torch.float64).cpu()
        bs = 1000
        while(start<end):
            inst_visibility[start:start+bs] = (proposals_pred[start:start+bs] / visibility.clip(min=1e-6)[None, :].cpu().to(torch.float64))
            start += bs
        torch.cuda.empty_cache()    
        proposals_pred[inst_visibility < cfg.cluster.point_visi] = 0
    else: # pointvis==0.0
        pass
    
    proposals_pred = proposals_pred.bool()

    if cfg.cluster.point_visi > 0:
        proposals_pred_final = custom_scatter_mean(
            proposals_pred,
            spp[None, :].expand(len(proposals_pred), -1),
            dim=-1,
            pool=True,
            output_type=torch.float64,
        )
        proposals_pred = (proposals_pred_final >= 0.5)[:, spp]

    ## Valid points
    mask_valid = proposals_pred.sum(1) > cfg.cluster.valid_points
    proposals_pred = proposals_pred[mask_valid].cpu()
    confidence = confidence[mask_valid].cpu()

    return proposals_pred, confidence
