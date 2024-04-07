# Based on https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
# Adapted by Ayca Takmaz, July 2023 OpenSUN3D Challenge
# Adapted for Reiplca by PhucNDA, Aug 2023 Open3DIS

import cv2
import numpy as np
import os
import torch

# def scaling_mapping(mapping, a, b, c, d):
#     # Calculate scaling factors
#     scale_x = c / a
#     scale_y = d / b

#     mapping[:, 0] = mapping[:, 0] * scale_x
#     mapping[:, 1] = mapping[:, 1] * scale_y
#     return mapping

class ReplicaReader(object):
    def __init__(
        self,
        root_path,
        cfg,
    ):
        """
        Args:
            class_names: list of str
            root_path: path with all info for a scene_id
                color, color_2det, depth, label, vote, ...
        """
        self.root_path = root_path

        self.scene_id = os.path.basename(root_path)

        # pipeline does box residual coding here

        depth_folder = os.path.join(self.root_path, "depth")
        if not os.path.exists(depth_folder):
            self.frame_ids = []
        else:
            depth_images = os.listdir(depth_folder)
            self.frame_ids = sorted([x.split(".")[0] for x in depth_images])

            # depth_images = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
            # self.frame_ids = [os.path.basename(x) for x in depth_images]
            # self.frame_ids = [x.split(".png")[0] for x in self.frame_ids]
            # self.video_id = depth_folder.split('/')[-2] #depth_folder.split('/')[-3]
            # self.frame_ids = [x for x in self.frame_ids]
            # self.frame_ids.sort()

        print("Number of original frames:", len(self.frame_ids))

        # get intrinsics
        self.global_intrinsic = np.array([
            [317.5, 0.0, 319.5],
            [0.0, 317.6471, 179.5],
            [0.0,0.0,1.0]
        ])

        self.depth_scale = 6553.5
        
    
        intrinsic_file = os.path.join(self.root_path, "intrinsic.txt")
        try:
            self.intrinsic = np.loadtxt(intrinsic_file)
        except:
            self.intrinsic = self.global_intrinsic

        # self.scene_pcd_path = os.path.join(cfg.data.original_ply, f"{self.scene_id}_vh_clean_2.ply")
        # video_path / f"{args.video}_vh_clean_2.ply"
        

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frame_ids)
    
    def read_depth(self, depth_path):

        depth_image = cv2.imread(depth_path, -1)
        depth_image = depth_image / self.depth_scale #rescale to obtain depth in meters

        return depth_image
    
    def read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
    
    def read_pose(self, pose_path):
        pose = np.loadtxt(pose_path)

        return pose
    
    def read_pointcloud(self, pcd_path=None):
        # if pcd_path is None:
        #     pcd_path = self.scene_pcd_path

        # try:
        #     scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
        #     point = np.array(scene_pcd.points)
        # except:
        # scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
        point, _, _, _ = torch.load(f"data/replica/replica_3d/{self.scene_id}.pth")
        
        return point
    
    def read_spp(self, spp_path, device='cuda'):
        spp = torch.load(spp_path)
        if isinstance(np.ndarray, spp):
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
        # fname = "{}.png".format(frame_id)
        depth_image_path = os.path.join(self.root_path, "depth", fnamedepth)
        image_path = os.path.join(self.root_path, "color", fnamecolor)
        pose_path = os.path.join(self.root_path, "pose", fnamepose)

        # if not os.path.exists(depth_image_path):
        #     print(depth_image_path)
        # if not os.path.exists(depth_image_path):
        #     print(depth_image_path, "does not exist")

        frame["depth_path"] = depth_image_path
        frame["image_path"] = image_path
        frame["pose_path"] = pose_path

        # frame["depth"] = cv2.imread(depth_image_path, -1)
        # frame["image"] = cv2.cvtColor(cv2.imread(image_path, ), cv2.COLOR_BGR2RGB)

        # depth_height, depth_width = frame["depth"].shape
        # im_height, im_width, im_channels = frame["image"].shape

        frame["intrinsics"] = self.intrinsic
        frame["global_intrinsic"] = self.global_intrinsic

        # if str(frame_id) in self.poses.keys():
        #     frame_pose = np.array(self.poses[str(frame_id)])
        # frame["pose"] = copy.deepcopy(frame_pose)

        # if depth_height != im_height:
        #     frame["image"] = np.zeros([depth_height, depth_width, 3])  # 288, 384, 3
        #     frame["image"][48 : 48 + 192, 64 : 64 + 256, :] = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # (m, n, _) = frame["image"].shape
        # depth_image = frame["depth"] / self.depth_scale #rescale to obtain depth in meters
        # rgb_image = frame["image"] / 255.0

        # pcd, rgb_feat = generate_point(
        #     rgb_image,
        #     depth_image,
        #     frame["intrinsics"],
        #     self.subsample,
        #     self.world_coordinate,
        #     frame_pose,
        # )

        # frame["pcd"] = None
        # frame["color"] = None
        # frame["depth"] = depth_image # depth in meters
        # frame["global_in"] = self.global_intrinsic
        return frame
