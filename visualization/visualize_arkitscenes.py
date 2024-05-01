import torch
import numpy as np
import random
import os

import pyviz3d.visualizer as viz
import random
from os.path import join
import open3d as o3d

def generate_palette(n):
    palette = []
    for _ in range(n):
        red = random.randint(0, 255)
        green = random.randint(0, 255)
        blue = random.randint(0, 255)
        palette.append((red, green, blue))
    return palette

def rle_decode(rle):
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

def read_pointcloud(pcd_path):
    scene_pcd = o3d.io.read_point_cloud(str(pcd_path))
    point = np.array(scene_pcd.points)
    color = np.array(scene_pcd.colors)

    return point, color


class VisualizationArkit:
    def __init__(self, point, color):
        self.point = point
        self.color = color
        self.vis = viz.Visualizer()
        self.vis.add_points(f'pcl', point, color.astype(np.float32), point_size=20, visible=True)
    
    def save(self, path):
        self.vis.save(path)
    
    def superpointviz(self, spp_path):
        print('...Visualizing Superpoints...')
        spp = torch.from_numpy(torch.load(spp_path)).to(device='cuda')
        unique_spp, spp, num_point = torch.unique(spp, return_inverse=True, return_counts=True)
        n_spp = unique_spp.shape[0]
        pallete =  generate_palette(n_spp + 1)
        uniqueness = torch.unique(spp).clone()
        # skip -1 
        tt_col = self.color.copy()
        for i in range(0, uniqueness.shape[0]):
            ss = torch.where(spp == uniqueness[i].item())[0]
            for ind in ss:
                tt_col[ind,:] = pallete[int(uniqueness[i].item())]
        self.vis.add_points(f'superpoint: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')
    
    # def vizmask3d(self, mask3d_path, specific = False):
    #     print('...Visualizing 3D backbone mask...')
    #     dic = torch.load(mask3d_path)
    #     instance = dic['ins']
    #     conf3d = dic['conf']
    #     pallete =  generate_palette(int(2e3 + 1))
    #     tt_col = self.color.copy()
    #     limit = 10
    #     for i in range(0, instance.shape[0]):
    #         tt_col[instance[i] == 1] = pallete[i]
    #         if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
    #             limit -= 1
    #             tt_col_specific = self.color.copy()
    #             tt_col_specific[instance[i] == 1] = pallete[i]
    #             self.vis.add_points(f'3D backbone mask: ' + str(i) + '_' + str(conf3d[i]), self.point, tt_col_specific, point_size=20, visible=True)

    #     self.vis.add_points(f'3D backbone mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
    #     print('---Done---')

    # def vizmask2d(self, mask2d_path, specific = False):
    #     print('...Visualizing 2D lifted mask...')
    #     dic = torch.load(mask2d_path)
    #     instance = dic['ins']
    #     instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
    #     conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)
    #     pallete =  generate_palette(int(5e3 + 1))
    #     tt_col = self.color.copy()
    #     limit = 10
    #     for i in range(0, instance.shape[0]):
    #         tt_col[instance[i] == 1] = pallete[i]
    #         if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
    #             limit -= 1
    #             tt_col_specific = self.color.copy()
    #             tt_col_specific[instance[i] == 1] = pallete[i]
    #             self.vis.add_points(f'2D lifted mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

    #     self.vis.add_points(f'2D lifted mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
    #     print('---Done---')        
        
    # def finalviz(self, agnostic_path, specific = False, vocab = False):
    #     print('...Visualizing final class agnostic mask...')
    #     dic = torch.load(agnostic_path)
    #     instance = dic['ins']
    #     instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
    #     conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)

    #     if vocab == True:
    #         label = dic['final_class']
    #     pallete =  generate_palette(int(2e3 + 1))
    #     tt_col = self.color.copy()
    #     limit = 5
    #     for i in range(0, instance.shape[0]):
    #         tt_col[instance[i] == 1] = pallete[i]
    #         if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
    #             limit -= 1
    #             tt_col_specific = self.color.copy()
    #             tt_col_specific[instance[i] == 1] = pallete[i]
    #             if vocab == True:
    #                 self.vis.add_points(f'final mask: ' + str(i) + '_' + class_names[label[i]], self.point, tt_col_specific, point_size=20, visible=True)                
    #             else:
    #                 self.vis.add_points(f'final mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

    #     self.vis.add_points(f'final mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
    #     print('---Done---')  

if __name__ == "__main__":
    
    '''
        Visualization using PyViz3D
        1. superpoint visualization
        
    
    '''
    # Scene ID to visualize
    scene_id = '42446478'

    ##### The format follows the dataset tree
    ## 1
    check_superpointviz = True
    spp_path = './data/ArkitDev/superpoints/' + scene_id + '.pth'

    pyviz3d_dir = '../viz' # visualization directory

    # Visualize Point Cloud 
    ply_file = './data/ArkitDev/original_ply_files'
    point, color = read_pointcloud(os.path.join(ply_file,scene_id + '_3dod_mesh.ply'))
    color = color * 127.5

    VIZ = VisualizationArkit(point, color)    
    
    if check_superpointviz:
        VIZ.superpointviz(spp_path)

    VIZ.save(pyviz3d_dir)
