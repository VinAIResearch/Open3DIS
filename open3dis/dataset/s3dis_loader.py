# Based on https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py
# Adapted by Ayca Takmaz, July 2023 OpenSUN3D Challenge
# Adapted for Reiplca by PhucNDA, Aug 2023 Open3DIS

import cv2
import numpy as np
import os
import torch
import glob
# def scaling_mapping(mapping, a, b, c, d):
#     # Calculate scaling factors
#     scale_x = c / a
#     scale_y = d / b

#     mapping[:, 0] = mapping[:, 0] * scale_x
#     mapping[:, 1] = mapping[:, 1] * scale_y
#     return mapping

AREA = 'Area_4'

class S3DISReader(object):
    def __init__(
        self,
        root_path,
        cfg,

        # scene_name,
        # # class_names=class_names,
        # root_path=None,
        # gt_path=None,
        # logger=None,
        # frame_rate=1,
        # with_color_image=True,
        # subsample=1,
        # world_coordinate=True,
        # time_dist_limit=0.3
    ):
        """
        Args:
            class_names: list of str
            root_path: path with all info for a scene_id
                color, color_2det, depth, label, vote, ...
            gt_path: xxx.json
                just to get correct floor height
            an2d_root: path to scene_id.json
                or None
            logger:
            frame_rate: int
            subsample: int
            world_coordinate: bool
        """
        # self.root_path = root_path

        self.root_path = root_path

        self.scene_id = os.path.basename(self.root_path)

        # self.area = AREA
        # scene_name.split('_')[0]
        # self.room = scene_name[7:]
        # self.root_path = f'../Dataset/s3dis/data_2d/{self.area}/{self.room}'

        # pipeline does box residual coding here
        # self.num_class = len(class_names)

        # self.dc = ARKitDatasetConfig()


        # rgb
        # breakpoint()
        depth_folder = os.path.join(self.root_path, "depth")
        if not os.path.exists(depth_folder):
            self.frame_ids = []
        else:
            depth_images = glob.glob(os.path.join(depth_folder, "*.png"))
            self.frame_ids = sorted([os.path.basename(x).split(".png")[0] for x in depth_images])
            # self.frame_ids = [x.split(".png")[0] for x in self.frame_ids]
            # self.video_id = depth_folder.split('/')[-2] #depth_folder.split('/')[-3]
            # self.frame_ids = [x for x in self.frame_ids]
            # self.frame_ids.sort()
            # self.intrinsics = {}
            self.global_intrinsic = None
        # breakpoint()
        print("Number of original frames:", len(self.frame_ids))

        # get intrinsics
        # self.global_intrinsic = np.loadtxt('../data/ScannetV2/ScannetV2_2D_5interval/trainval/intrinsic_depth.txt')
        # self.poses = {}
        # # intrinsic_file = os.path.join(self.root_path, "intrinsic.txt")
        # pose_folder = os.path.join(self.root_path, "pose")
        # for frame_id in self.frame_ids:
        #     cam_pose = np.load(os.path.join(pose_folder, frame_id.replace('depth', 'pose') + '.npz'))
        #     self.intrinsics[frame_id] = cam_pose['intrinsic']
        #     self.poses[frame_id] = np.linalg.inv(cam_pose['pose'])
            # self.intrinsics[frame_id] = np.loadtxt(intrinsic_file)
            # self.poses[frame_id] = np.loadtxt(os.path.join(pose_folder, frame_id + '.txt'))

        # self.frame_rate = frame_rate
        # self.subsample = subsample
        # self.with_color_image = with_color_image
        # self.world_coordinate = world_coordinate

        self.depth_scale = 512.0

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
        cam_pose_dict = np.load(pose_path)
        
        return np.linalg.inv(cam_pose_dict['pose'])
    
    def read_pointcloud(self, pcd_path=None):
        # if pcd_path is None:
        #     pcd_path = self.scene_pcd_path

        # try:
        #     scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
        #     point = np.array(scene_pcd.points)
        # except:
        # scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
        scene_pcd_path = os.path.join('./data/s3dis/preprocess_open3dis', f"{AREA}_{self.scene_id}.pth")
        point, _, _, _ = torch.load(scene_pcd_path)

        n_points = len(point)
        if n_points > 1000000:
            stride = 8
        elif n_points >= 600000:
            stride = 6
        elif n_points >= 400000:
            stride = 2
        else:
            stride = 1
        point = point[::stride]

        return point
    

    def read_spp(self, spp_path, device='cuda'):
        name = os.path.basename(spp_path)
        dir_path = os.path.dirname(spp_path)
        new_spp_path = os.path.join(dir_path, f'{AREA}_{name}')
        spp = torch.load(new_spp_path)
        if isinstance(spp, np.ndarray):
            spp = torch.from_numpy(spp)

        n_points = len(spp)
        if n_points > 1000000:
            stride = 8
        elif n_points >= 600000:
            stride = 6
        elif n_points >= 400000:
            stride = 2
        else:
            stride = 1
        spp = spp[::stride]

        spp = spp.to(device)

        return spp
    
    def read_feature(self, feat_path, device='cuda'):
        name = os.path.basename(feat_path)
        dir_path = os.path.dirname(feat_path)
        new_feat_path = os.path.join(dir_path, f'{AREA}_{name}')

        dc_feature = torch.load(new_feat_path)
        if isinstance(dc_feature, np.ndarray):
            dc_feature = torch.from_numpy(dc_feature)

        n_points = len(dc_feature)
        if n_points > 1000000:
            stride = 8
        elif n_points >= 600000:
            stride = 6
        elif n_points >= 400000:
            stride = 2
        else:
            stride = 1
        dc_feature = dc_feature[::stride]
        
        dc_feature = dc_feature.to(device)
        return dc_feature

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
        fnamecolor = "{}.png".format(frame_id.replace('depth', 'rgb'))
        fnamepose = "{}.npz".format(frame_id.replace('depth', 'pose'))
        # fname = "{}.png".format(frame_id)
        depth_image_path = os.path.join(self.root_path, "depth", fnamedepth)
        image_path = os.path.join(self.root_path, "rgb", fnamecolor)
        pose_path = os.path.join(self.root_path, "pose", fnamepose)

        # if not os.path.exists(depth_image_path):
        #     print(depth_image_path, "does not exist")
        # frame["depth"] = cv2.imread(depth_image_path, -1)

        # print(image_path, depth_image_path)
        # frame["image"] = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # frame["image_path"] = fnamecolor
        # depth_height, depth_width = frame["depth"].shape
        # im_height, im_width, im_channels = frame["image"].shape


        frame["depth_path"] = depth_image_path
        frame["image_path"] = image_path
        frame["pose_path"] = pose_path

        # frame["intrinsics"] = self.intrinsic
        frame["global_intrinsic"] = None

        frame["intrinsics"] = np.load(pose_path)['intrinsic']
            # self.poses[frame_id] = np.linalg.inv(cam_pose['pose'])
        
        # frame["intrinsics"] = copy.deepcopy(self.intrinsics[frame_id])

        # if str(frame_id) in self.poses.keys():
        #     frame_pose = np.array(self.poses[str(frame_id)])
        # frame["pose"] = copy.deepcopy(frame_pose)

        # if depth_height != im_height:
        #     frame["image"] = np.zeros([depth_height, depth_width, 3])  # 288, 384, 3
        #     frame["image"][48 : 48 + 192, 64 : 64 + 256, :] = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # (m, n, _) = frame["image"].shape
        # depth_image = frame["depth"] / 512.0 #rescale to obtain depth in meters
        # rgb_image = frame["image"] / 255.0

        # pcd, rgb_feat = generate_point(
        #     rgb_image,
        #     depth_image,
        #     frame["intrinsics"],
        #     self.subsample,
        #     self.world_coordinate,
        #     frame_pose,
        # )
        # global_in = np.eye(4)
        # global_in[:3, :3] = frame["intrinsics"]
        # frame["pcd"] = None
        # frame["color"] = None
        # frame["depth"] = depth_image # depth in meters
        # frame["global_in"] = global_in
        return frame