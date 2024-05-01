# Based on https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
# Adapted by Ayca Takmaz, July 2023 OpenSUN3D Challenge
# Adapted for Scannet by PhucNDA, Aug 2023 Open3DIS

import os

import cv2
import numpy as np
import open3d as o3d
import torch

def scaling_mapping(mapping, a, b, c, d):
    # Scale mapping value from depth image to RGB image by interpolation
    # Calculate scaling factors
    scale_x = c / a
    scale_y = d / b

    mapping[:, 0] = mapping[:, 0] * scale_x
    mapping[:, 1] = mapping[:, 1] * scale_y
    return mapping


class ScanNetReader(object):
    def __init__(
        self,
        root_path,
        cfg,
    ):
        self.root_path = root_path
        self.scene_id = os.path.basename(root_path)

        # pipeline does box residual coding here

        depth_folder = os.path.join(self.root_path, "depth")
        if not os.path.exists(depth_folder):
            self.frame_ids = []
        else:
            depth_images = os.listdir(depth_folder)
            self.frame_ids = sorted([x.split(".")[0] for x in depth_images])

        print("Number of original frames:", len(self.frame_ids))

        # get intrinsics only in Scannet
        self.global_intrinsic = np.array(
            [[571.623718, 0.0, 319.5], 
             [0.0, 571.623718, 239.5],
             [0.0, 0.0, 1.0]])
             
        self.depth_scale = 1000.0
        
        intrinsic_file = os.path.join(self.root_path, "intrinsic.txt")
        try:
            self.intrinsic = np.loadtxt(intrinsic_file)
        except:
            self.intrinsic = None
            print('No global intrinsic')

        self.scene_pcd_path = os.path.join(cfg.data.original_ply, f"{self.scene_id}.ply")

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frame_ids)

    def read_depth(self, depth_path):
        depth_image = cv2.imread(depth_path, -1)
        depth_image = depth_image / self.depth_scale  # rescale to obtain depth in meters
        return depth_image

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read_pose(self, pose_path):
        pose = np.loadtxt(pose_path)
        return pose

    def read_pointcloud(self, pcd_path=None):
        if pcd_path is None:
            pcd_path = self.scene_pcd_path
        scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
        point = np.array(scene_pcd.points)

        return point
    
    def read_gt_3D(self, gt_path):
        _, _, sem_gt, inst_gt = torch.load(gt_path)
        return sem_gt, inst_gt
    
    def read_spp(self, spp_path, device='cuda'):
        spp = torch.load(spp_path)
        if isinstance(spp, np.ndarray):
            spp = torch.from_numpy(spp)
        spp = spp.to(device)

        return spp
    
    def read_feature(self, feat_path, device='cuda'):
        dc_feature = torch.load(feat_path)
        if isinstance(dc_feature, np.ndarray):
            dc_feature = torch.from_numpy(dc_feature)
        
        dc_feature = dc_feature.to(device)
        return dc_feature
    
    def read_3D_proposal(self, agnostic3d_path):
        agnostic3d_data = torch.load(agnostic3d_path)
        return agnostic3d_data
    

    def __getitem__(self, idx):
        """
        Returns:
            frame: a dict
                {frame_id}: str
                {depth}: (h, w)
                {image}: (h, w)
                {image_path}: str
                {intrinsics}: np.array 3x3
                {pose}: np.array 4x4
                {pcd}: np.array (n, 3)
                    in world coordinate
                {color}: (n, 3)
        """
        frame_id = self.frame_ids[idx]
        frame = {}
        frame["frame_id"] = frame_id
        fnamedepth = "{}.png".format(frame_id)
        fnamecolor = "{}.jpg".format(frame_id)
        fnamepose = "{}.txt".format(frame_id)
        fnameintrinsic = "{}.txt".format(frame_id)

        depth_image_path = os.path.join(self.root_path, "depth", fnamedepth)
        image_path = os.path.join(self.root_path, "color", fnamecolor)
        pose_path = os.path.join(self.root_path, "pose", fnamepose)
        
        
        frame["depth_path"] = depth_image_path
        frame["image_path"] = image_path
        frame["pose_path"] = pose_path

        frame["intrinsics"] = self.intrinsic
        frame["global_intrinsic"] = self.global_intrinsic
        frame["scannet_depth_intrinsic"] = np.array(
            [[577.870605, 0.0, 319.5], 
             [0.0, 577.870605, 239.5],
             [0.0, 0.0, 1.0]])
        try:
            frame_intrinsic = os.path.join(self.root_path, "intrinsic", fnameintrinsic)
            frame["translated_intrinsics"] = np.loadtxt(frame_intrinsic) # for scannetpp only
        except:
            pass

        return frame
