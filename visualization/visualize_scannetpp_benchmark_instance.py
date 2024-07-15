import torch
import numpy as np
import random
import os

import pyviz3d.visualizer as viz
import random
from os.path import join
import open3d as o3d
from open3dis.dataset.scannetpp import SEMANTIC_CAT_SCANNET_PP, SEMANTIC_INSTANCE_CAT_SCANNET_PP, INSTANCE_BENCHMARK84_SCANNET_PP

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

# class_names = SEMANTIC_CAT_SCANNET_PP # original files
class_names = INSTANCE_BENCHMARK84_SCANNET_PP # modified Scannet200 like 
class_names = SEMANTIC_INSTANCE_BENCHMARK84_SCANNET_PP = [
    'wall', 'ceiling', 'floor', 'window frame', 'doorframe', 'pipe', 'windowsill', 'shower wall', 'kitchen counter', 'electrical duct', 'paper', 'board', 'cloth', 'blind rail', 'air vent', 'clothes hanger', 
    'table', 'door', 'ceiling lamp', 'cabinet', 'blinds', 'curtain', 'chair', 'storage cabinet', 'office chair', 'bookshelf', 'whiteboard', 'window', 'box', 'monitor', 'shelf', 'heater', 'kitchen cabinet', 'sofa', 'bed', 'trash can', 'book', 'plant', 'blanket', 'tv', 'computer tower', 'refrigerator', 'jacket', 'sink', 'bag', 'picture', 'pillow', 'towel', 'suitcase', 'backpack', 'crate', 'keyboard', 'rack', 'toilet', 'printer', 'poster', 'painting', 'microwave', 'shoes', 'socket', 'bottle', 'bucket', 'cushion', 'basket', 'shoe rack', 'telephone', 'file folder', 'laptop', 'plant pot', 'exhaust fan', 'cup', 'coat hanger', 'light switch', 'speaker', 'table lamp', 'kettle', 'smoke detector', 'container', 'power strip', 'slippers', 'paper bag', 'mouse', 'cutting board', 'toilet paper', 'paper towel', 'pot', 'clock', 'pan', 'tap', 'jar', 'soap dispenser', 'binder', 'bowl', 'tissue box', 'whiteboard eraser', 'toilet brush', 'spray bottle', 'headphones', 'stapler', 'marker'
]


class VisualizationScannetpp:
    def __init__(self, point, color):
        self.point = point
        self.color = color
        
        self.vis = viz.Visualizer()
        self.vis.add_points(f'pcl', point, color.astype(np.float32), point_size=20, visible=True)

    
    def save(self, path):
        self.vis.save(path)
    
    def superpointviz(self, spp_path):
        print('...Visualizing Superpoints...')
        tt_col = self.color.copy()
        spp = torch.from_numpy(torch.load(spp_path)).to(device='cuda')
        unique_spp, spp, num_point = torch.unique(spp, return_inverse=True, return_counts=True)
        n_spp = unique_spp.shape[0]
        pallete =  generate_palette(n_spp + 1)
        uniqueness = torch.unique(spp).clone()
        # skip -1 
        for i in range(0, uniqueness.shape[0]):
            ss = torch.where(spp == uniqueness[i].item())[0]
            for ind in ss:
                tt_col[ind,:] = pallete[int(uniqueness[i].item())]
        self.vis.add_points(f'superpoint: ' + str(i), self.point, tt_col, point_size=20, visible=True)

        print('---Done---')
    
    def gtviz(self, gt_data, specific = True): # pending
        print('...Visualizing Groundtruth...')
        normalized_point, normalized_color, sem_label, ins_label = torch.load(gt_data)
        pallete =  generate_palette(int(7e3 + 1))
        n_label = np.unique(ins_label)
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, n_label.shape[0]):
            if n_label[i] == -100:
                continue
            tt_col[np.where(ins_label==n_label[i])] = pallete[i]
            limit -= 1
            if specific and limit > 0: # be more specific
                tt_col_specific = self.color.copy()
                tt_col_specific[np.where(ins_label==n_label[i])] = pallete[i]
                self.vis.add_points(f'GT instance: ' + str(i) + '_' + class_names[sem_label[np.where(ins_label==n_label[i])][0]], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'GT instance: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')

    def vizmask3d(self, mask3d_path, specific = False):
        print('...Visualizing 3D backbone mask...')
        dic = torch.load(mask3d_path)
        instance = dic['ins']
        try:
            instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        except:
            pass
        conf3d = dic['conf']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()

        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(f'3D backbone mask: ' + str(i) + '_' + str(conf3d[i]), self.point, tt_col_specific, point_size=20, visible=True)
        self.vis.add_points(f'3D backbone mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')

    def vizmask2d(self, mask2d_path, specific = False):
        print('...Visualizing 2D lifted mask...')
        dic = torch.load(mask2d_path)
        instance = dic['ins']
        instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)
        pallete =  generate_palette(int(6e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                self.vis.add_points(f'2D lifted mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'2D lifted mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')        
        
    def finalviz(self, agnostic_path, specific = False, vocab = False):
        print('...Visualizing final class agnostic mask...')
        dic = torch.load(agnostic_path)
        instance = dic['ins']
        instance = torch.stack([torch.tensor(rle_decode(ins)) for ins in instance])
        conf2d = dic['conf'] # confidence really doesn't affect much (large mask -> small conf)

        if vocab == True:
            label = dic['final_class']
        pallete =  generate_palette(int(2e3 + 1))
        tt_col = self.color.copy()
        limit = 10
        for i in range(0, instance.shape[0]):
            tt_col[instance[i] == 1] = pallete[i]
            if specific and limit > 0: # be more specific but limit 10 masks (avoiding lag)
                limit -= 1
                tt_col_specific = self.color.copy()
                tt_col_specific[instance[i] == 1] = pallete[i]
                if vocab == True:
                    self.vis.add_points(f'final mask: ' + str(i) + '_' + class_names[label[i]], self.point, tt_col_specific, point_size=20, visible=True)                
                else:
                    self.vis.add_points(f'final mask: ' + str(i) + '_' + str(conf2d[i].item())[:5], self.point, tt_col_specific, point_size=20, visible=True)

        self.vis.add_points(f'final mask: ' + str(i), self.point, tt_col, point_size=20, visible=True)
        print('---Done---')  
        

if __name__ == "__main__":
    
    '''
        Visualization using PyViz3D
        1. superpoint visualization
        2. ground-truth annotation
        3. 3D backbone mask (isbnet, mask3d) -- class-agnostic
        4. lifted 2D masks -- class-agnostic
        5. final masks --class-agnostic (2D+3D) | vocab
    
    '''
    # Scene ID to visualize
    scene_id = '0a5c013435'

    ##### The format follows the dataset tree
    ## 1
    check_superpointviz = False
    spp_path = './data/Scannetpp/Scannetpp_3D/test/superpoints/' + scene_id + '.pth'
    ## 2
    check_gtviz = True
    gt_path = './data/Scannetpp/Scannetpp_3D/train/groundtruth_benchmark_instance/' + scene_id + '.pth'
    ## 3
    check_3dviz = False
    mask3d_path = './data/Scannetpp/Scannetpp_3D/test/isbnet_clsagnostic_scannetpp/' + scene_id + '.pth'
    ## 4
    check_2dviz = False
    mask2d_path = '../exp_scannetpp/version_benchmarkinstance_groundedsam/hier_agglo/' + scene_id + '.pth'
    ## 5
    check_finalviz = False
    agnostic_path = '../exp_scannetpp/version_benchmarkinstance_groundedsam/final_result_hier_agglo/' + scene_id + '.pth'
    

    pyviz3d_dir = '../viz' # visualization directory

    # Visualize Point Cloud 
    ply_file = './data/Scannetpp/Scannetpp_3D/train/original_ply_files'
    point, color = read_pointcloud(os.path.join(ply_file,scene_id + '.ply'))
    color = color * 127.5
    
    VIZ = VisualizationScannetpp(point, color)    
    
    if check_superpointviz:
        VIZ.superpointviz(spp_path)
    if check_gtviz:
        VIZ.gtviz(gt_path, specific = True)
    if check_3dviz:
        VIZ.vizmask3d(mask3d_path, specific = False)
    if check_2dviz:
        VIZ.vizmask2d(mask2d_path, specific = False)
    if check_finalviz:
        VIZ.finalviz(agnostic_path, specific = True, vocab = True)
    VIZ.save(pyviz3d_dir)
