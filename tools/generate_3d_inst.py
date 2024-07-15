import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import clip
import torch
import yaml
from munch import Munch
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200 # Scannet200
from open3dis.dataset.scannetpp import SEMANTIC_CAT_SCANNET_PP, INSTANCE_BENCHMARK84_SCANNET_PP # ScannetPP
from open3dis.dataset.replica import INSTANCE_CAT_REPLICA
from open3dis.dataset.s3dis import INSTANCE_CAT_S3DIS, AREA

from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval
from open3dis.src.clustering.clustering import process_hierarchical_agglomerative_spp, process_hierarchical_agglomerative_nospp
from torch.nn import functional as F
from tqdm import tqdm, trange


def rle_encode_gpu_batch(masks):
    """
    Encode RLE (Run-length-encode) from 1D binary mask.
    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1

        runs[1::2] -= runs[::2]

        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles


def rle_decode(rle):
    """
    Decode rle to get binary mask.
    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle["length"]
    try:
        s = rle["counts"].split()
    except:
        s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask


def get_final_instances(
    cfg, text_features, cluster_dict=None, use_2d_proposals=False, use_3d_proposals=True, only_instance=True
):
    """
    Get final 3D instance (2D | 3D), point cloud features from which stage
    returning masks, class and scores
    """
    exp_path = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name)

    pc_features_path = os.path.join(exp_path, cfg.exp.grounded_feat_output, f"{scene_id}.pth")
    pc_refined_features_path = os.path.join(exp_path, cfg.exp.refined_grounded_feat_output, f"{scene_id}.pth")
    


    # 2D lifting 3D mask path
    cluster_dict_path = os.path.join(exp_path, cfg.exp.clustering_3d_output, f"{scene_id}.pth")

    if cluster_dict is not None:
        data = cluster_dict
    else:
        data = torch.load(cluster_dict_path)

    if isinstance(data["ins"][0], dict):
        instance_2d = torch.stack([torch.from_numpy(rle_decode(ins)) for ins in data["ins"]], dim=0).cuda()
    else:
        instance_2d = data["ins"].cuda()

    confidence_2d = torch.tensor(data["conf"]).cuda()

    ########### Proposal branch selection ###########
    if use_3d_proposals:
        if cfg.data.dataset_name == 's3dis':
            agnostic3d_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path, f"{AREA}_{scene_id}.pth")
        else:
            agnostic3d_path = os.path.join(cfg.data.cls_agnostic_3d_proposals_path, f"{scene_id}.pth")
        agnostic3d_data = torch.load(agnostic3d_path)
        instance_3d_encoded = np.array(agnostic3d_data["ins"])
        confidence_3d = torch.tensor(agnostic3d_data["conf"]).cuda()

        n_instance_3d = instance_3d_encoded.shape[0]

        if isinstance(instance_3d_encoded[0], dict):
            instance_3d = torch.stack(
                [torch.from_numpy(rle_decode(in3d)) for in3d in instance_3d_encoded], dim=0
            ).cuda()
        else:
            instance_3d = torch.stack([torch.tensor(in3d) for in3d in instance_3d_encoded], dim=0).cuda()

    if use_2d_proposals and use_3d_proposals:
        instance = torch.cat([instance_2d, instance_3d], dim=0)
        confidence = torch.cat([confidence_2d, confidence_3d], dim=0)
    elif use_2d_proposals and not use_3d_proposals:
        instance = instance_2d
        confidence = torch.cat([confidence_2d], dim=0)
    else:
        instance = instance_3d
        confidence = torch.cat([confidence_3d], dim=0)
    ########### ########### ########### ###########

    n_instance = instance.shape[0]

    if only_instance == True:  # Return class-agnostic 3D instance
        return instance, None, None

    # Choose which stage to use the feature ?
    if cfg.proposals.refined and os.path.exists(pc_refined_features_path):
        pc_features = torch.load(pc_refined_features_path)["feat"].cuda()
    else:
        pc_features = torch.load(pc_features_path)["feat"].cuda()
    
    pc_features = F.normalize(pc_features, dim=1, p=2)    
    
    ### Offloading CPU for scannetpp @@
    # NOTE Pointwise semantic scores
    # predicted_class = (cfg.final_instance.scale_semantic_score * pc_features @ text_features.cuda().T.float()).softmax(dim=-1)

    predicted_class = torch.zeros((pc_features.shape[0], text_features.shape[0]), dtype = torch.float32)
    bs = 100000
    for batch in range(0, pc_features.shape[0], bs):
        start = batch
        end = min(start + bs, pc_features.shape[0])
        predicted_class[start:end] = (cfg.final_instance.scale_semantic_score * pc_features[start:end].cpu() @ text_features.T.cpu().to(torch.float32)).softmax(dim=-1).cpu()
    predicted_class = predicted_class.cuda()

    del pc_features
    torch.cuda.empty_cache() 
    # NOTE Mask-wise semantic scores
    inst_class_scores = torch.einsum("kn,nc->kc", instance.float().cpu(), predicted_class.float().cpu()).cuda()  # K x classes
    inst_class_scores = inst_class_scores / instance.float().cuda().sum(dim=1)[:, None]  # K x classes




    if cfg.final_instance.duplicate:
        # # NOTE Top-K instances
        inst_class_scores = inst_class_scores.reshape(-1)  # n_cls * n_queries

        labels = (
            torch.arange(cfg.data.num_classes, device=inst_class_scores.device)
            .unsqueeze(0)
            .repeat(n_instance, 1)
            .flatten(0, 1)
        )
        cur_topk = 600 if use_3d_proposals else cfg.final_instance.top_k
        _, idx = torch.topk(inst_class_scores, k=min(cur_topk, len(inst_class_scores)), largest=True)
        mask_idx = torch.div(idx, cfg.data.num_classes, rounding_mode="floor")

        cls_final = labels[idx]
        scores_final = inst_class_scores[idx].cuda()
        masks_final = instance[mask_idx]
    else:
        idx = torch.argmax(inst_class_scores, dim = -1)
        cls_final = idx.cpu()
        scores_final = inst_class_scores[:, idx]
        masks_final = instance


    return masks_final, cls_final, scores_final


# evaluate_openvocab = False
# evaluate_agnostic = False

class DeticMask:
    def __init__(self, pred_masks_rle=None, scores=None, pred_masks=None):
        self.pred_masks_rle = pred_masks_rle
        self.scores = scores
        self.pred_masks = pred_masks

def get_parser():
    parser = argparse.ArgumentParser(description="Configuration Open3DIS")
    parser.add_argument("--config",type=str,required = True,help="Config")
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    cfg = Munch.fromDict(yaml.safe_load(open(args.config, "r").read()))

    evaluate_openvocab = cfg.evaluate.evalvocab  # Evaluation for openvocab
    evaluate_agnostic = cfg.evaluate.evalagnostic  # Evaluation for openvocab


    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    # Scannet200 class text features saving
    if cfg.data.dataset_name == 'scannet200':
        class_names = INSTANCE_CAT_SCANNET_200
    elif cfg.data.dataset_name == 'scannetpp':
        class_names = SEMANTIC_CAT_SCANNET_PP    
    elif cfg.data.dataset_name == 'replica':
        class_names = INSTANCE_CAT_REPLICA
    elif cfg.data.dataset_name == 's3dis':     
        class_names = INSTANCE_CAT_S3DIS
    elif cfg.data.dataset_name == 'scannetpp_benchmark':
        class_names = INSTANCE_BENCHMARK84_SCANNET_PP
    elif cfg.data.dataset_name == 'arkitscenes':
        class_names = ['None', 'None']
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

    if evaluate_openvocab:
        scan_eval = ScanNetEval(class_labels=class_names, dataset_name=cfg.data.dataset_name)
        gtsem = []
        gtinst = []
        res = []
        
    clip_adapter, clip_preprocess = clip.load(cfg.foundation_model.clip_model, device = 'cuda')

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = clip_adapter.encode_text(clip.tokenize(class_names).cuda())
        text_features /= text_features.norm(dim=-1, keepdim=True)
    # Prepare directories
    save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    os.makedirs(save_dir_cluster, exist_ok=True)
    save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output) # final_output
    os.makedirs(save_dir_final, exist_ok=True)

    # Multiprocess logger
    if os.path.exists("tracker_lifted.txt") == False:
        with open("tracker_lifted.txt", "w") as file:
            file.write("Processed Scenes .\n")

    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        for scene_id in tqdm(scene_ids):
            print("Process", scene_id)
            ## Tracker
            done = False
            path = scene_id + ".pth"
            with open("tracker_lifted.txt", "r") as file:
                lines = file.readlines()
                lines = [line.strip() for line in lines]
                for line in lines:
                    if path in line:
                        done = True
                        break
            if done == True:
                print("existed " + path)
                continue
            ## Write append each line
            with open("tracker_lifted.txt", "a") as file:
                file.write(path + "\n")

            #############################################
            # NOTE hierarchical agglomerative clustering
            if True:
                cluster_dict = None
                if cfg.final_instance.spp_level: # Group by Superpoints
                    proposals3d, confidence = process_hierarchical_agglomerative_spp(scene_id, cfg)
                else:
                    proposals3d, confidence = process_hierarchical_agglomerative_nospp(scene_id, cfg)

                if proposals3d == None: # Discarding too large scene
                    continue

                cluster_dict = {
                    "ins": rle_encode_gpu_batch(proposals3d),
                    "conf": confidence,
                }
                torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))

            #############################################
            # NOTE get final instances
            if False:   
                cluster_dict = torch.load(os.path.join(save_dir_cluster, f"{scene_id}.pth"))
                masks_final, cls_final, scores_final = get_final_instances(
                    cfg,
                    text_features,
                    cluster_dict=cluster_dict,
                    use_2d_proposals=cfg.proposals.p2d,
                    use_3d_proposals=cfg.proposals.p3d,
                    only_instance=cfg.proposals.agnostic,
                )
                if scores_final == None:
                    final_dict = {
                        "ins": rle_encode_gpu_batch(masks_final),
                        "conf": None,
                        "final_class": None,
                    }
                else:
                    final_dict = {
                        "ins": rle_encode_gpu_batch(masks_final),
                        "conf": scores_final.cpu(),
                        "final_class": cls_final.cpu(),
                    }
                # NOTE Final instance
                torch.save(final_dict, os.path.join(save_dir_final, f"{scene_id}.pth"))
            #############################################
            # NOTE Evaluation openvocab
            if evaluate_openvocab:
                
                if cfg.data.dataset_name == 's3dis':
                    gt_path = os.path.join(cfg.data.gt_pth, f"{AREA}_{scene_id}.pth")
                    _, _, sem_gt, inst_gt = torch.load(gt_path)
                    n_points = len(sem_gt)
                    if n_points > 1000000:
                        stride = 8
                    elif n_points >= 600000:
                        stride = 6
                    elif n_points >= 400000:
                        stride = 2
                    else:
                        stride = 1
                    sem_gt = sem_gt[::stride]
                    inst_gt = inst_gt[::stride]

                    # NOTE do not eval class "clutter"
                    inst_gt[sem_gt==12] = -100
                    sem_gt[sem_gt==12] = -100
                else:
                    gt_path = os.path.join(cfg.data.gt_pth, f"{scene_id}.pth")
                    _, _, sem_gt, inst_gt = torch.load(gt_path)

                gtsem.append(np.array(sem_gt).astype(np.int32))
                gtinst.append(np.array(inst_gt).astype(np.int32))

                masks_final = masks_final.cpu()
                cls_final = cls_final.cpu()

                n_mask = masks_final.shape[0]
                tmp = []
                for ind in range(n_mask):
                    mask = (masks_final[ind] == 1).numpy().astype(np.uint8)
                    conf = 1.0  # Same as OpenMask3D
                    final_class = float(cls_final[ind])
                    tmp.append({"scan_id": scene_id, "label_id": final_class + 1, "conf": conf, "pred_mask": mask})
                res.append(tmp)
            # NOTE Evaluation agnostic
            if evaluate_agnostic:
                pass

            print("Done")
            torch.cuda.empty_cache()

        if evaluate_openvocab:
            scan_eval.evaluate(
                res, gtsem, gtinst, exp_path=os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output)
            )
        if evaluate_agnostic:
            pass
