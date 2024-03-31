import os
import re

import numpy as np
import torch
import torch_scatter


class PointCloudToImageMapper(object):
    def __init__(
        self, image_dim, visibility_threshold=0.1, cut_bound=0, intrinsics=None, device="cpu", use_torch=False
    ):

        self.image_dim = image_dim
        self.vis_thres = visibility_threshold
        self.cut_bound = cut_bound
        self.intrinsics = intrinsics

        self.device = device
        if use_torch:
            self.intrinsics = torch.from_numpy(self.intrinsics).to(device)

    def compute_mapping_torch(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        device = coords.device
        if intrinsic is not None: # adjust intrinsic
            self.intrinsics = intrinsic
        else:
            intrinsic = self.intrinsics
        camera_to_world = torch.from_numpy(camera_to_world).to(device).float()

        mapping = torch.zeros((3, coords.shape[0]), dtype=torch.long, device=device)
        coords_new = torch.cat([coords, torch.ones([coords.shape[0], 1], dtype=torch.float, device=device)], dim=1).T

        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.linalg.inv(camera_to_world)
        p = world_to_camera.float() @ coords_new.float()
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = torch.round(p).long()  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth = torch.from_numpy(depth).to(device)
            occlusion_mask = torch.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= self.vis_thres
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask

        new_inside_mask = inside_mask

        mapping[0][new_inside_mask] = pi[1][new_inside_mask]
        mapping[1][new_inside_mask] = pi[0][new_inside_mask]
        mapping[2][new_inside_mask] = 1

        return mapping.T

    def compute_mapping(self, camera_to_world, coords, depth=None, intrinsic=None):
        """
        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        if self.intrinsics is not None:  # global intrinsics
            intrinsic = self.intrinsics
        mapping = np.zeros((3, coords.shape[0]), dtype=int)
        coords_new = np.concatenate([coords, np.ones([coords.shape[0], 1])], axis=1).T
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = np.linalg.inv(camera_to_world)
        p = np.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = np.round(p).astype(np.int32)  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )
        if depth is not None:
            depth_cur = depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = (
                np.abs(depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]) <= self.vis_thres * depth_cur
            )
            inside_mask[inside_mask == True] = occlusion_mask
        else:
            front_mask = p[2] > 0  # make sure the depth is in front
            inside_mask = front_mask * inside_mask

        # NOTE detect occlusion
        pi_x_ = pi[1][inside_mask]
        pi_y_ = pi[0][inside_mask]
        pi_depth_ = pi[2][inside_mask]

        inds = (pi_x_ * self.image_dim[0] + pi_y_).astype(np.int32)
        _, inds = np.unique(inds, return_inverse=True)

        depth_min = torch_scatter.scatter_min(
            torch.from_numpy(pi_depth_).float(), torch.from_numpy(inds).long(), dim=0
        )[0]
        depth_min = torch.where(depth_min < 0.0, 0.0, depth_min)
        depth_min = depth_min.numpy()
        depth_min_broadcast = depth_min[inds]

        THRESHOLD = 0.2  # (meter)
        depth_occlusion_mask = (pi_depth_ - depth_min_broadcast) <= THRESHOLD

        new_inside_mask = inside_mask.copy()
        new_inside_mask[inside_mask] = depth_occlusion_mask
        ############################

        mapping[0][new_inside_mask] = pi[1][new_inside_mask]
        mapping[1][new_inside_mask] = pi[0][new_inside_mask]
        mapping[2][new_inside_mask] = 1

        return mapping.T
