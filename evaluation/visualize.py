import numpy as np
import torch
# Point mapper
import pyviz3d.visualizer as viz
from PIL import Image, ImageDraw
##############################################

if __name__ == "__main__":
    scene_id = 'scene0568_00'
    path = '../../Dataset/ScannetV2/ScannetV2_2D_5interval/trainval/version1/final_sem'
    pred_label = torch.load(os.path.join(path, scene_id + '.pth'))
    data_type = 'trainval'
    gt_data = '../../Dataset/ScannetV2/ScannetV2_2D_5interval/' + data_type + '/pcl/'+scene_id+'_inst_nostuff.pth'
    
    point, color, sem_label, ins_label = torch.load(gt_data)
    if True: 
        v = viz.Visualizer()
        color = (color + 1) * 127.5
        # There are 2 query class
        sem_class = np.unique(pred_label)
        v.add_points(f'pcl color', point, color, point_size=20, visible=True)
        for ind in sem_class:
            tmp = color.copy()
            tmp[torch.where(pred_label==ind)[0]]=np.array((0,255,0))
            v.add_points(f'sem_'+str(ind), point, tmp, point_size=20, visible=True)
        v.save('viz')


    

