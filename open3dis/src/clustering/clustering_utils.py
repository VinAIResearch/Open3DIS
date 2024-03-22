import argparse
import copy
import glob
import importlib
import json
import operator
import os
import pickle
import sys
from collections import deque
from functools import reduce
from pathlib import Path
from time import perf_counter
from typing import Dict, Union

import numpy as np
import numpy.linalg as la
import pycocotools.mask
import torch
import torch.nn.functional as F
import torch_scatter
from detectron2.structures import Instances
from numba import njit
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm import tqdm


def custom_scatter_mean(input_feats, indices, dim=0, pool=True, output_type=None):
    if not pool:
        return input_feats

    original_type = input_feats.dtype
    with torch.cuda.amp.autocast(enabled=False):
        out_feats = torch_scatter.scatter_mean(input_feats.to(torch.float32), indices, dim=dim)

    if output_type is None:
        out_feats = out_feats.to(original_type)
    else:
        out_feats = out_feats.to(output_type)

    return out_feats


def resolve_overlapping_3d_masks(pred_masks, pred_scores, score_thresh=0.5, device="cuda:0"):
    M, N = pred_masks.shape
    # panoptic_masks = torch.clone(pred_masks)
    scores = torch.from_numpy(pred_scores)[:, None].repeat(1, N)
    scores[~pred_masks] = 0

    panoptic_masks = torch.argmax(scores, dim=0)
    return panoptic_masks


def resolve_overlapping_masks(pred_masks, pred_scores, score_thresh=0.5, device="cuda:0"):
    M, H, W = pred_masks.shape
    pred_masks = torch.from_numpy(pred_masks).to(device)
    panoptic_masks = torch.clone(pred_masks)
    scores = torch.from_numpy(pred_scores)[:, None, None].repeat(1, H, W).to(device)
    scores[~pred_masks] = 0
    indices = ((scores == torch.max(scores, dim=0, keepdim=True).values) & pred_masks).nonzero()
    panoptic_masks = torch.zeros((M, H, W), dtype=torch.bool, device=device)
    panoptic_masks[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    panoptic_masks[scores > score_thresh] = True  # if prediction score is high enough, keep the mask anyway

    # return panoptic_masks

    return panoptic_masks.detach().cpu().numpy()


def read_detectron_instances(filepath: Union[str, os.PathLike], rle_to_mask=True) -> Instances:
    with open(filepath, "rb") as fp:
        instances = pickle.load(fp)
        if rle_to_mask:
            if instances.pred_masks_rle:
                pred_masks = np.stack([pycocotools.mask.decode(rle) for rle in instances.pred_masks_rle])
                instances.pred_masks = torch.from_numpy(pred_masks).to(torch.bool)  # (M, H, W)
            else:
                instances.pred_masks = torch.empty((0, 0, 0), dtype=torch.bool)
    return instances


def compute_projected_pts_torch(pts, cam_intr, device="cuda"):
    N = pts.shape[0]
    projected_pts = torch.zeros((N, 2), dtype=torch.int64, device=device)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]

    z = pts[:, 2]
    projected_pts[:, 0] = torch.round(fx * pts[:, 0] / z + cx)
    projected_pts[:, 1] = torch.round(fy * pts[:, 1] / z + cy)
    return projected_pts


@njit
def compute_projected_pts(pts, cam_intr):
    N = pts.shape[0]
    projected_pts = np.empty((N, 2), dtype=np.int64)
    fx, fy = cam_intr[0, 0], cam_intr[1, 1]
    cx, cy = cam_intr[0, 2], cam_intr[1, 2]
    for i in range(pts.shape[0]):
        z = pts[i, 2]
        x = int(np.round(fx * pts[i, 0] / z + cx))
        y = int(np.round(fy * pts[i, 1] / z + cy))
        projected_pts[i, 0] = x
        projected_pts[i, 1] = y
    return projected_pts


def compute_visibility_mask_torch(pts, projected_pts, depth_im, depth_thresh=0.005, device="cuda"):
    im_h, im_w = depth_im.shape
    visibility_mask = torch.zeros(projected_pts.shape[0], dtype=torch.bool, device=device)

    z = pts[:, 2]
    x, y = projected_pts[:, 0], projected_pts[:, 1]

    cond = (
        (x >= 0)
        & (y < im_w)
        & (y >= 0)
        & (y < im_h)
        & (depth_im[y, x] > 0)
        & (torch.abs(z - depth_im[y, x]) < depth_thresh)
    )
    visibility_mask[cond] = 1
    return visibility_mask


@njit
def compute_visibility_mask(pts, projected_pts, depth_im, depth_thresh=0.005):
    im_h, im_w = depth_im.shape
    visibility_mask = np.zeros(projected_pts.shape[0]).astype(np.bool8)
    for i in range(projected_pts.shape[0]):
        x, y = projected_pts[i]
        z = pts[i, 2]
        if x < 0 or x >= im_w or y < 0 or y >= im_h:
            continue
        if depth_im[y, x] == 0:
            continue
        if np.abs(z - depth_im[y, x]) < depth_thresh:
            visibility_mask[i] = True
    return visibility_mask


def compute_visible_masked_pts_torch(scene_pts, projected_pts, visibility_mask, pred_masks, device="cuda"):
    N = scene_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    masked_pts = torch.zeros((M, N), dtype=torch.bool, device=device)
    visible_indices = torch.nonzero(visibility_mask).view(-1)

    x_arr, y_arr = projected_pts[visible_indices, 0], projected_pts[visible_indices, 1]

    masked_pts[:, visible_indices] = pred_masks[:, y_arr, x_arr]
    # m_ind, y_ind, x_ind = torch.nonzero(pred_masks, as_tuple=True)
    return masked_pts
    pred_masks[y_arr, x_arr]

    for m in range(M):
        for i in visible_indices:
            x, y = projected_pts[i]
            if pred_masks[m, y, x]:
                masked_pts[m, i] = True
    return masked_pts


@njit
def compute_visible_masked_pts(scene_pts, projected_pts, visibility_mask, pred_masks):
    N = scene_pts.shape[0]
    M, _, _ = pred_masks.shape  # (M, H, W)
    masked_pts = np.zeros((M, N), dtype=np.bool_)
    visible_indices = np.nonzero(visibility_mask)[0]
    for m in range(M):
        for i in visible_indices:
            x, y = projected_pts[i]
            if pred_masks[m, y, x]:
                masked_pts[m, i] = True
    return masked_pts


def compute_relation_matrix_self(instance_pt_mask, spp, sieve):
    if not torch.is_tensor(instance_pt_mask):
        instance_pt_mask = torch.from_numpy(instance_pt_mask)
    torch.cuda.empty_cache()

    #### Small tweak make it work on scannetpp ~ 40GB A100
    # n = instance_pt_count.shape[1]
    # numbers = list(range(n))
    # chosen_numbers = random.sample(numbers, n // max(1,int(((n *instance_pt_count.shape[0])/1e8))))
    # instance_pt_mask = instance_pt_count[:,chosen_numbers].to(torch.bool).to(torch.float16)

    # torch.cuda.empty_cache()
    # intersection = []
    # for i in range(instance_pt_mask.shape[0]):
    #     it = []
    #     for j in range(instance_pt_mask.shape[0]):
    #         it.append(instance_pt_mask[i].cuda() @ instance_pt_mask.T[:, j].cuda())
    #         torch.cuda.empty_cache()
    #     intersection.append(torch.tensor(it))  # save mem
    # intersection = torch.stack(intersection).cuda()
    # (1k,1M) ~ 1e9
    instance_pt_mask_tmp = (instance_pt_mask.to(torch.float64) * sieve.expand(instance_pt_mask.shape[0], -1).to(torch.float64).cuda()).to(torch.float64)
    intersection =  (instance_pt_mask.to(torch.float64) @ instance_pt_mask_tmp.T.to(torch.float64)).to(torch.float64)
    inliers = instance_pt_mask_tmp.sum(1, keepdims=True).to(torch.float64).cuda()
    union = (inliers + inliers.T - intersection).to(torch.float64)
    iou_matrix = intersection / (union + 1e-6)
    precision_matrix = intersection / (inliers.T + 1e-6)
    recall_matrix = intersection / (inliers + 1e-6)
    torch.cuda.empty_cache()
    return iou_matrix.to(torch.float64), precision_matrix, recall_matrix.to(torch.float64)


def find_connected_components(adj_matrix):
    if torch.is_tensor(adj_matrix):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "adjacency matrix should be a square matrix"

    N = adj_matrix.shape[0]
    clusters = []
    visited = np.zeros(N, dtype=np.bool_)
    for i in range(N):
        if visited[i]:
            continue
        cluster = []
        queue = deque([i])
        visited[i] = True
        while queue:
            j = queue.popleft()
            cluster.append(j)
            for k in np.nonzero(adj_matrix[j])[0]:
                if not visited[k]:
                    queue.append(k)
                    visited[k] = True
        clusters.append(cluster)
    return clusters
