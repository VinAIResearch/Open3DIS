import numpy as np
import torch
from scannetv2_inst_eval import ScanNetEval
import os
from tqdm import tqdm
from isbnet.util.rle import rle_decode
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200


scan_eval = ScanNetEval(class_labels = INSTANCE_CAT_SCANNET_200)
data_path = "../exp/version_test/final_result_hier_agglo"
pcl_path = '../Dataset/Scannet200/val'

if __name__ == '__main__':
    scenes = sorted([s for s in os.listdir(data_path) if s.endswith('.pth')])

    gtsem = []
    gtinst = []
    res = []

    for scene in tqdm(scenes):

        gt_path = os.path.join(pcl_path, scene.replace('.pth','') + '_inst_nostuff.pth')
        loader = torch.load(gt_path)

        sem_gt, inst_gt = loader[2], loader[3]
        gtsem.append(np.array(sem_gt).astype(np.int32))
        gtinst.append(np.array(inst_gt).astype(np.int32))


        scene_path = os.path.join(data_path, scene)
        pred_mask = torch.load(scene_path)

        masks, category, score = pred_mask['ins'], pred_mask['final_class'], pred_mask['conf']

        n_mask = category.shape[0]
        tmp = []
        for ind in range(n_mask):
            if isinstance(masks[ind], dict):
                mask = rle_decode(masks[ind])
            else:
                mask = (masks[ind] == 1).numpy().astype(np.uint8)
            # conf = score[ind] #
            conf = 1.0
            final_class = float(category[ind])
            scene_id = scene.replace('.pth', '')
            tmp.append({'scan_id': scene_id, 'label_id':final_class + 1, 'conf':conf, 'pred_mask':mask})

        res.append(tmp)

    scan_eval.evaluate(res, gtsem, gtinst)



