import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import torch
import yaml
from munch import Munch
from open3dis.dataset.scannet200 import INSTANCE_CAT_SCANNET_200
from open3dis.evaluation.scannetv2_inst_eval import ScanNetEval
from open3dis.src.clustering.clustering import process_hierarchical_agglomerative
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

    # Choose which stage to use the feature ?
    if cfg.proposals.refined == True:
        pc_features_path = os.path.join(exp_path, cfg.exp.refined_grounded_feat_output, f"{scene_id}.pth")
    else:
        pc_features_path = os.path.join(exp_path, cfg.exp.grounded_feat_output, f"{scene_id}.pth")

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

        intersection = torch.einsum("nc,mc->nm", instance_2d.float(), instance_3d.float())
        # print(intersection.shape, instance.shape, )
        ious = intersection / (instance_2d.sum(1)[:, None] + instance_3d.sum(1)[None, :] - intersection)
        ious_max = torch.max(ious, dim=1)[0]

        valid_mask = torch.ones(instance_2d.shape[0], dtype=torch.bool, device=instance_2d.device)
        valid_mask[ious_max >= cfg.final_instance.iou_overlap] = 0

        instance_2d = instance_2d[valid_mask]
        confidence_2d = confidence_2d[valid_mask]

        instance = torch.cat([instance_2d, instance_3d], dim=0)
        confidence = torch.cat([confidence_2d, confidence_3d], dim=0)
    else:
        instance = instance_2d
        confidence = confidence_2d

    if use_2d_proposals and use_3d_proposals:
        instance = torch.cat([instance_2d, instance_3d], dim=0)
        confidence = torch.cat([confidence_2d, confidence_3d], dim=0)
    elif use_2d_proposals and not use_3d_proposals:
        instance = torch.cat([instance_2d], dim=0)
        confidence = torch.cat([confidence_2d], dim=0)
    else:
        instance = torch.cat([instance_3d], dim=0)
        confidence = torch.cat([confidence_3d], dim=0)
    ########### ########### ########### ###########

    n_instance = instance.shape[0]

    if only_instance == True:  # Return class-agnostic 3D instance
        return instance, None, None

    pc_features = torch.load(pc_features_path)["feat"].cuda().half()
    pc_features = F.normalize(pc_features, dim=1, p=2)

    # NOTE Pointwise semantic scores
    predicted_class = (cfg.final_instance.scale_semantic_score * pc_features @ text_features.T).softmax(dim=-1)

    # NOTE Mask-wise semantic scores
    inst_class_scores = torch.einsum("kn,nc->kc", instance.float(), predicted_class)  # K x classes
    inst_class_scores = inst_class_scores / instance.float().sum(dim=1)[:, None]  # K x classes

    # NOTE Top-K instances
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
    scores_final = inst_class_scores[idx]
    masks_final = instance[mask_idx]

    return masks_final, cls_final, scores_final


evaluate_openvocab = False
evaluate_agnostic = False

if __name__ == "__main__":

    cfg = Munch.fromDict(yaml.safe_load(open("./configs/scannet200.yaml", "r").read()))

    evaluate_openvocab = cfg.evaluate.evalvocab  # Evaluation for openvocab
    evaluate_agnostic = cfg.evaluate.evalagnostic  # Evaluation for openvocab

    with open(cfg.data.split_path, "r") as file:
        scene_ids = sorted([line.rstrip("\n") for line in file])

    # Scannet200 class text features saving
    class_names = INSTANCE_CAT_SCANNET_200
    if os.path.exists("./pretrains/text_features/scannet200_text_features.pth"):
        text_features = torch.load("./pretrains/text_features/scannet200_text_features.pth").cuda().half()
    else:
        clip_adapter, _, clip_preprocess = open_clip.create_model_and_transforms(
            cfg.foundation_model.clip_model, pretrained=cfg.foundation_model.clip_checkpoint
        )
        clip_adapter = clip_adapter.cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = clip_adapter.encode_text(open_clip.tokenize(class_names).cuda())
            text_features /= text_features.norm(dim=-1, keepdim=True)
            torch.save(text_features.cpu(), "./pretrains/text_features/scannet200_text_features.pth")

    # Prepare directories
    save_dir_cluster = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.clustering_3d_output)
    os.makedirs(save_dir_cluster, exist_ok=True)
    save_dir_final = os.path.join(cfg.exp.save_dir, cfg.exp.exp_name, cfg.exp.final_output)
    os.makedirs(save_dir_final, exist_ok=True)

    with torch.cuda.amp.autocast(enabled=cfg.fp16):
        if evaluate_openvocab:
            scan_eval = ScanNetEval(class_labels=INSTANCE_CAT_SCANNET_200)
            gtsem = []
            gtinst = []
            res = []
        if evaluate_agnostic:
            pass  # not yet

        for scene_id in tqdm(scene_ids):
            print("Process", scene_id)

            #############################################
            # NOTE hierarchical agglomerative clustering
            if True:
                breakpoint()
                cluster_dict = None
                proposals3d, confidence = process_hierarchical_agglomerative(scene_id, cfg)
                cluster_dict = {
                    "ins": rle_encode_gpu_batch(proposals3d),
                    "conf": confidence,
                }
                torch.save(cluster_dict, os.path.join(save_dir_cluster, f"{scene_id}.pth"))
            cluster_dict = torch.load(os.path.join(save_dir_cluster, f"{scene_id}.pth"))
            #############################################
            # NOTE get final instances
            masks_final, cls_final, scores_final = get_final_instances(
                cfg,
                text_features,
                cluster_dict=cluster_dict,
                use_2d_proposals=cfg.proposals.p2d,
                use_3d_proposals=cfg.proposals.p3d,
                only_instance=cfg.proposals.agnostic,
            )

            final_dict = {
                "ins": rle_encode_gpu_batch(masks_final),
                "conf": scores_final.cpu(),
                "final_class": cls_final.cpu(),
            }
            # Final instance
            torch.save(final_dict, os.path.join(save_dir_final, f"{scene_id}.pth"))
            #############################################

            # NOTE Evaluation openvocab
            if evaluate_openvocab:
                gt_path = os.path.join(cfg.data.gt_pth, f"{scene_id}_inst_nostuff.pth")
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
