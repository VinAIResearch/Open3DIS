# ArkitScenesReader
# Based on https://github.com/apple/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py

# Adapted by Ayca Takmaz, July 2023 OpenSUN3D challenge
# Adapted by PhucNDA, May 2024  Open3DIS

import copy
import cv2
import glob
import json
import numpy as np
import bisect
import os
import pdb
import math
import numpy as np
import open3d as o3d
import torch

class ARKitDatasetConfig(object):
    def __init__(self):
        pass

def eulerAnglesToRotationMatrix(theta):
    """Euler rotation matrix with clockwise logic.
    Rotation

    Args:
        theta: list of float
            [theta_x, theta_y, theta_z]
    Returns:
        R: np.array (3, 3)
            rotation matrix of Rz*Ry*Rx
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def upright_camera_relative_transform(pose):
    """Generate pose matrix with z-dim as height

    Args:
        pose: np.array (4, 4)
    Returns:
        urc: (4, 4)
        urc_inv: (4, 4)
    """

    # take viewing direction in camera local coordiantes (which is simply unit vector along +z)
    view_dir_camera = np.asarray([0, 0, 1])
    R = pose[0:3, 0:3]
    t = pose[0:3, 3]

    # convert to world coordinates
    view_dir_world = np.dot(R, view_dir_camera)

    # compute heading
    view_dir_xy = view_dir_world[0:2]
    heading = math.atan2(view_dir_xy[1], view_dir_xy[0])

    # compute rotation around Z to align heading with +Y
    zRot = -heading + math.pi / 2

    # translation first, back to camera point
    urc_t = np.identity(4)
    urc_t[0:2, 3] = -1 * t[0:2]

    # compute rotation matrix
    urc_r = np.identity(4)
    urc_r[0:3, 0:3] = eulerAnglesToRotationMatrix([0, 0, zRot])

    urc = np.dot(urc_r, urc_t)
    urc_inv = np.linalg.inv(urc)

    return urc, urc_inv

def rotate_pc(pc, rotmat):
    """Rotation points w.r.t. rotmat
    Args:
        pc: np.array (n, 3)
        rotmat: np.array (4, 4)
    Returns:
        pc: (n, 3)
    """
    pc_4 = np.ones([pc.shape[0], 4])
    pc_4[:, 0:3] = pc
    pc_4 = np.dot(pc_4, np.transpose(rotmat))

    return pc_4[:, 0:3]

def rotate_points_along_z(points, angle):
    """Rotation clockwise
    Args:
        points: np.array of np.array (B, N, 3 + C) or
            (N, 3 + C) for single batch
        angle: np.array of np.array (B, )
            or (, ) for single batch
            angle along z-axis, angle increases x ==> y
    Returns:
        points_rot:  (B, N, 3 + C) or (N, 3 + C)

    """
    single_batch = len(points.shape) == 2
    if single_batch:
        points = np.expand_dims(points, axis=0)
        angle = np.expand_dims(angle, axis=0)
    cosa = np.expand_dims(np.cos(angle), axis=1)
    sina = np.expand_dims(np.sin(angle), axis=1)
    zeros = np.zeros_like(cosa) # angle.new_zeros(points.shape[0])
    ones = np.ones_like(sina) # angle.new_ones(points.shape[0])

    rot_matrix = (
        np.concatenate((cosa, -sina, zeros, sina, cosa, zeros, zeros, zeros, ones), axis=1)
        .reshape(-1, 3, 3)
    )

    # print(rot_matrix.view(3, 3))
    points_rot = np.matmul(points[:, :, :3], rot_matrix)
    points_rot = np.concatenate((points_rot, points[:, :, 3:]), axis=-1)

    if single_batch:
        points_rot = points_rot.squeeze(0)

    return points_rot

def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis.
    Arguments:
        angle_axis {Point3} -- a rotation in angle axis form.
    """
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix

def TrajStringToMatrix(traj_str):
    """ convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


class ArkitScenesReader(object):
    def __init__(
        self,
        root_path=None,
        cfg = None,
        logger=None,
        frame_rate=1,
        with_color_image=True,
        subsample=1,
        world_coordinate=True,
        time_dist_limit=0.3
    ):
        """
        Args:
            class_names: list of str
            root_path: path with all info for a scene_id
                color, color_2det, depth, label, vote, ...
            an2d_root: path to scene_id.json
                or None
            logger:
            frame_rate: int
            subsample: int
            world_coordinate: bool
        """
        self.root_path = root_path
        self.scene_id = os.path.basename(root_path)

        # pipeline does box residual coding here

        self.dc = ARKitDatasetConfig()

        depth_folder = os.path.join(self.root_path, "lowres_depth")
        if not os.path.exists(depth_folder):
            self.frame_ids = []
        else:
            depth_images = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
            self.frame_ids = [os.path.basename(x) for x in depth_images]
            self.frame_ids = [x.split(".png")[0].split("_")[1] for x in self.frame_ids]
            self.video_id = depth_folder.split('/')[-2] #depth_folder.split('/')[-3]
            self.frame_ids = [x for x in self.frame_ids]
            self.frame_ids.sort()
            self.intrinsics = {}

        traj_file = os.path.join(self.root_path, 'lowres_wide.traj')
        with open(traj_file) as f:
            self.traj = f.readlines()
        # convert traj to json dict
        poses_from_traj = {}
        for line in self.traj:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = TrajStringToMatrix(line)[1].tolist()

        if os.path.exists(traj_file):
            # self.poses = json.load(open(traj_file))
            self.poses = poses_from_traj
        else:
            self.poses = {}

        self.frame_ids_new, self.closest_poses = find_closest_pose_from_timestamp(self.frame_ids, list(self.poses.keys()), time_dist_limit=time_dist_limit)
        print("Number of original frames:", len(self.frame_ids))
        print("Number of frames with poses:", len(self.frame_ids_new)) # 4682 frames to 4663 frames
        assert len(self.frame_ids_new) == len(self.closest_poses)

        self.frame_ids = self.frame_ids_new

        # get intrinsics
        for frame_id in self.frame_ids:
            intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics", f"{self.video_id}_{frame_id}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{float(frame_id) - 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(self.root_path, "lowres_wide_intrinsics",
                                            f"{self.video_id}_{float(frame_id) + 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                print("frame_id", frame_id)
                print(intrinsic_fn)
            self.intrinsics[frame_id] = st2_camera_intrinsics(intrinsic_fn)

        # get intrinsics only in Scannet
        self.global_intrinsic = np.array(
            [[571.623718, 0.0, 319.5], 
             [0.0, 571.623718, 239.5],
             [0.0, 0.0, 1.0]])
        
        self.depth_scale = 1000.0
        self.frame_rate = frame_rate
        self.subsample = subsample
        self.with_color_image = with_color_image
        self.world_coordinate = world_coordinate
        self.scene_pcd_path = os.path.join(cfg.data.original_ply, f"{self.scene_id}_3dod_mesh.ply")


    def __iter__(self):
        return self

    def __len__(self):
        return len(self.frame_ids)

    def read_feature(self, feat_path, device='cuda'):
        dc_feature = torch.load(feat_path)
        if isinstance(dc_feature, np.ndarray):
            dc_feature = torch.from_numpy(dc_feature)
        
        dc_feature = dc_feature.to(device)
        return dc_feature
    
    def read_3D_proposal(self, agnostic3d_path):
        agnostic3d_data = torch.load(agnostic3d_path)
        return agnostic3d_data

    def read_depth(self, depth_path):
        depth_image = cv2.imread(depth_path, -1)
        depth_image = depth_image / self.depth_scale  # rescale to obtain depth in meters
        return depth_image

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read_pose(self, pose_path):
        return pose_path

    def read_pointcloud(self, pcd_path=None):
        if pcd_path is None:
            pcd_path = self.scene_pcd_path
        scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
        point = np.array(scene_pcd.points)

        return point
    
    def read_spp(self, spp_path, device='cuda'):
        spp = torch.load(spp_path)
        if isinstance(spp, np.ndarray):
            spp = torch.from_numpy(spp)
        spp = spp.to(device)

        return spp

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
                {color}: (n, 3)
        """
        frame_id = self.frame_ids[idx]
        frame = {}
        frame["frame_id"] = frame_id
        fname = "{}_{}.png".format(self.video_id, frame_id)
        # fname = "{}.png".format(frame_id)
        depth_image_path = os.path.join(self.root_path, "lowres_depth", fname)
        if not os.path.exists(depth_image_path):
            print(depth_image_path)

        image_path = os.path.join(self.root_path, "lowres_wide", fname)

        if not os.path.exists(depth_image_path):
            print(depth_image_path, "does not exist")
        frame["depth"] = cv2.imread(depth_image_path, -1)
        frame["depth_path"] = depth_image_path

        frame["image"] = cv2.cvtColor(cv2.imread(image_path, ), cv2.COLOR_BGR2RGB)

        frame["image_path"] = image_path
        depth_height, depth_width = frame["depth"].shape
        im_height, im_width, im_channels = frame["image"].shape

        frame["intrinsics"] = copy.deepcopy(self.intrinsics[frame_id])

        if str(frame_id) in self.poses.keys():
            frame_pose = np.array(self.poses[str(frame_id)])
        else:
            frame_pose = np.array(self.poses[self.closest_poses[idx]])
        frame["pose"] = copy.deepcopy(frame_pose)
        frame["pose_path"] = copy.deepcopy(frame_pose)
        
        if depth_height != im_height:
            frame["image"] = np.zeros([depth_height, depth_width, 3])  # 288, 384, 3
            frame["image"][48 : 48 + 192, 64 : 64 + 256, :] = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        (m, n, _) = frame["image"].shape
        depth_image = frame["depth"] / 1000.0 #rescale to obtain depth in meters
        rgb_image = frame["image"] / 255.0

        frame["depth"] = depth_image # depth in meters
        return frame


def find_closest_pose_from_timestamp(image_timestamps, pose_timestamps, time_dist_limit=0.3):
    closest_poses = []
    new_frame_ids = []

    for image_ts in image_timestamps:
        index = bisect.bisect_left(pose_timestamps, image_ts)
        
        if index == 0:
            closest_pose = pose_timestamps[index]
        elif index == len(pose_timestamps):
            closest_pose = pose_timestamps[index - 1]
        else:
            diff_prev = abs(float(image_ts) - float(pose_timestamps[index - 1]))
            diff_next = abs(float(image_ts) - float(pose_timestamps[index]))
            
            if diff_prev < diff_next:
                closest_pose = pose_timestamps[index - 1]
            else:
                closest_pose = pose_timestamps[index]

        if abs(float(closest_pose) - float(image_ts))>=time_dist_limit:
            pass
            #print("Warning: Closest pose is not close enough to image timestamp:", image_ts, closest_pose)
        else:
            closest_poses.append(closest_pose)
            new_frame_ids.append(image_ts)

    return new_frame_ids, closest_poses


