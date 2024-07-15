# NOT DONE
import torch
import numpy as np
import os
import cv2
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

#### Foundation 2D
import clip
from util2d.openai_clip import CLIP_OpenAI
from util2d.segment_anything_hq import SAM_HQ
import groundingdino.datasets.transforms as T

#### YOLO-World
from detectron2.structures import BitMasks
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms

#### RAM Plus
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
#### Util
import supervision as sv
from util2d.util import show_mask, masks_to_rle 
#### Open3DIS util
from open3dis.dataset import build_dataset
from open3dis.dataset.scannet_loader import ScanNetReader, scaling_mapping
from open3dis.src.fusion_util import NMS_cuda
from open3dis.src.mapper import PointCloudToImageMapper



class RAM_YOLOWorld_SAM:
    ###################################################################
    #                        RAM YOLO-World + SAM                     #
    ###################################################################
    def __init__(self, cfg):
        # Load Foundation Model
        sam2d = SAM_HQ(cfg)
        clip2d = CLIP_OpenAI(cfg)
        self.sam_predictor = sam2d.sam_predictor
        self.clip_adapter, self.clip_preprocess = clip2d.clip_adapter, clip2d.clip_preprocess
        # Load RAM++
        self.ram_transform, self.ram_model = self.init_ram(cfg)

        # Load YOLO-World model
        self.runner_yoloworld_model = self.init_segmenter2d_models(cfg)

    def gen_grounded_mask_and_feat(self, scene_id, class_names, cfg, gen_feat=True):
        """
        RAM + YOLO World + SAM, CLIP
            Generate 2D masks from YoloWorld Ram box prompt
            Accmulate CLIP mask feature onto 3D point cloud
        Returning boxes and logits scores for each chunk in the caption with box & text threshoding
        """
        scene_dir = os.path.join(cfg.data.datapath, scene_id)

        loader = build_dataset(root_path=scene_dir, cfg=cfg)
        # scannet_loader = ScanNetReader(root_path=scene_dir, cfg=cfg)

        # Pointcloud Image mapper
        img_dim = cfg.data.img_dim
        pointcloud_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=loader.global_intrinsic, cut_bound=cfg.data.cut_num_pixel_boundary
        )

        points = loader.read_pointcloud()
        points = torch.from_numpy(points).cuda()
        n_points = points.shape[0]

        grounded_data_dict = {}

        # Accmulate CLIP mask feature onto 3D point cloud ?
        if gen_feat:
            grounded_features = torch.zeros((n_points, cfg.foundation_model.clip_dim)).cuda()
        else:
            grounded_features = None

        for i in trange(0, len(loader), cfg.data.img_interval):
            frame = loader[i]
            frame_id = frame["frame_id"]  # str
            image_path = frame["image_path"]  # str

            #### Processing Grounding DINO ####
            image_pil = Image.open(image_path).convert("RGB")
            image_pil, image_infer = self.load_image(image_pil)

            boxes_filt = []
            confs_filt = []
            labels_filt = []

            #### Processing RAM++ ####
            image = self.ram_transform(image_pil).unsqueeze(0).to('cuda')
            res = inference(image, self.ram_model)
            res = res[0].split('|')
            res = [st.strip() for st in res]

            ### RAM querying into multiple chunks -- see Supplementary
            segment_size = 2 # scannetpp_benchmark
            segments = [res[i : i + segment_size] for i in range(0, len(res), segment_size)]
            for cls_name in segments:
                boxes, labels, confidences = self.get_grounding_output(
                    image_path,
                    ".".join(cls_name),
                    device=cfg.foundation_model.device,
                )

                if len(boxes) > 0:
                    boxes_filt.append(torch.tensor(boxes))
                    confs_filt.append(torch.tensor(confidences))
                    for name in labels:
                        labels_filt.append(name)
            if len(boxes_filt) == 0:  # No box in that view
                continue
            boxes_filt = torch.cat(boxes_filt)
            confs_filt = torch.cat(confs_filt)

            size = image_pil.size
            H, W = size[1], size[0]
            boxes_filt = boxes_filt.cuda()

            # default openyolo XYXY
            l, t, r, b = (
                boxes_filt[:, 0].clip(0),
                boxes_filt[:, 1].clip(0),
                boxes_filt[:, 2].clip(min=0, max=W),
                boxes_filt[:, 3].clip(min=0, max=H),
            )

            # Filtering big boxes
            valid_boxes = ((b - t) > 1) & ((r - l) > 1) & ((b - t) * (r - l) / (W * H) < 0.85)
            target_id_valid = torch.nonzero(valid_boxes).view(-1)

            if len(target_id_valid) == 0:  # No valid box
                continue
            boxes_filt = boxes_filt[target_id_valid]
            confs_filt = confs_filt[target_id_valid]
            labels_filt = [labels_filt[id] for id in target_id_valid]

            #### Segment Anything ####
            image_sam = cv2.imread(image_path)
            image_sam = cv2.cvtColor(image_sam, cv2.COLOR_BGR2RGB)
            rgb_img_dim = image_sam.shape[:2]
            self.sam_predictor.set_image(image_sam)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image_sam.shape[:2])  # .to(device)
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to(cfg.foundation_model.device),
                multimask_output=False,
            )

            if masks == None:  # No mask in the view
                continue

            masks_fitted = torch.zeros_like(masks, dtype=bool)
            regions = []

            for box_id, box in enumerate(boxes_filt):
                l, t, r, b = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                l = max(l, 0)
                t = max(t, 0)
                r = min(r, W)
                b = min(b, H)
                # Outercrop 2D mask
                masks_fitted[box_id, 0, t:b, l:r] = True
                row, col = torch.where(masks[box_id][0, t:b, l:r] == False)
                tmp = torch.tensor(image_sam)[t:b, l:r, :].cuda()
                # Blurring background - trick here improve CLIP feature
                tmp[row, col, 0] = (0 * 0.5 + tmp[row, col, 0] * (1 - 0.5)).to(torch.uint8)
                tmp[row, col, 1] = (0 * 0.5 + tmp[row, col, 1] * (1 - 0.5)).to(torch.uint8)
                tmp[row, col, 2] = (0 * 0.5 + tmp[row, col, 2] * (1 - 0.5)).to(torch.uint8)
        
                regions.append(self.clip_preprocess(Image.fromarray((tmp.cpu().numpy()))))

            masks = torch.logical_and(masks, masks_fitted)  # fitting
            imgs = torch.stack(regions).cuda()
            img_batches = torch.split(imgs, 64, dim=0)
            image_features = []

            # Batch forwarding CLIP
            with torch.no_grad(), torch.cuda.amp.autocast():
                for img_batch in img_batches:
                    image_feat = self.clip_adapter.encode_image(img_batch)
                    image_feat /= image_feat.norm(dim=-1, keepdim=True)
                    image_features.append(image_feat)
            image_features = torch.cat(image_features, dim=0)
            if False:
                # draw output image
                image = loader.read_image(image_path)
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                for mask in masks:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                plt.axis("off")
                # plot out
                os.makedirs("../debug/scannet200/" + scene_id, exist_ok=True)
                plt.savefig(
                    os.path.join("../debug/scannet200/" + scene_id + "/openyolosam_" + str(i) + ".jpg"),
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=0.0,
                )
            
            #### SAVING MASKS, CLIP FEATURES
            grounded_data_dict[frame_id] = {
                "boxes": boxes_filt.cpu(),
                "masks": masks_to_rle(masks),
                "class": labels_filt,
                "img_feat": image_features.cpu(),
                "conf": confs_filt.cpu(),
            }
            if gen_feat:
                pose = loader.read_pose(frame["pose_path"])
                depth = loader.read_depth(frame["depth_path"])
                
                if "scannetpp" in cfg.data.dataset_name:  # Map on image resolution in Scannetpp only
                    depth = cv2.resize(depth, (img_dim[0], img_dim[1]))
                    mapping = torch.ones([n_points, 4], dtype=int, device="cuda")
                    mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic=frame["translated_intrinsics"])

                elif "scannet200" in cfg.data.dataset_name:
                    mapping = torch.ones([n_points, 4], dtype=int, device=points.device)
                    mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic = frame["scannet_depth_intrinsic"])
                    new_mapping = scaling_mapping(
                        torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], rgb_img_dim[0], rgb_img_dim[1]
                    )
                    mapping[:, 1:4] = torch.cat((new_mapping, mapping[:, 3].unsqueeze(1)), dim=1)

                elif "replica" in cfg.data.dataset_name:
                    mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
                    mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth)

                elif "s3dis" in cfg.data.dataset_name:
                    mapping = torch.ones([n_points, 4], dtype=int, device='cuda')
                    mapping[:, 1:4] = pointcloud_mapper.compute_mapping_torch(pose, points, depth, intrinsic=frame["intrinsics"])

                else:
                    raise ValueError(f"Unknown dataset: {cfg.data.dataset_name}")

                idx = torch.where(mapping[:, 3] == 1)[0]

                if False: # Visualize highlighted points
                    import pyviz3d.visualizer as viz
                    image = loader.read_image(image_path)
                    for tmp in mapping[idx]:
                        x, y = tmp[1].item(), tmp[2].item()
                        image = cv2.circle(image, (y,x), radius=0, color=(0, 0, 255), thickness=-5)
                    cv2.imwrite('../test.png', image)
                    vis = viz.Visualizer()
                    color = torch.zeros_like(points).cpu().numpy()
                    color[idx.cpu(),0] =  255
                    vis.add_points(f'pcl', points.cpu().numpy(), color, point_size=20, visible=True)
                    vis.save('../viz')

                if len(idx) < 100:  # No points corresponds to this image, visible points on 2D image
                    continue

                pred_masks = BitMasks(masks.squeeze(1))
                # Flood fill single CLIP feature for 2D mask
                final_feat = torch.einsum("qc,qhw->chw", image_features.float(), pred_masks.tensor.float())
                ### Summing features
                grounded_features[idx] += final_feat[:, mapping[idx, 1], mapping[idx, 2]].permute(1, 0)

        grounded_features = grounded_features.cpu()
        return grounded_data_dict, grounded_features

    def get_grounding_output(self, image, caption, with_logits=True, device="cuda"):
        """
        YOLO-World box generator
        Returning boxes and logits scores for each chunk in the caption with box & text threshoding
        """

        # Caption formatting
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        
        texts = [[t.strip()] for t in caption.split(".")[:-1]] + [[" "]]
        data_info = self.runner_yoloworld_model.pipeline(dict(img_id=0, img_path=image, texts=texts))

        data_batch = dict(
            inputs=data_info["inputs"].unsqueeze(0),
            data_samples=[data_info["data_samples"]],
        )

        with autocast(enabled=False), torch.no_grad():
            output = self.runner_yoloworld_model.model.test_step(data_batch)[0]
            self.runner_yoloworld_model.model.class_names = texts
            pred_instances = output.pred_instances

        # nms
        keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=0.5)
        pred_instances = pred_instances[keep_idxs]
        pred_instances = pred_instances[pred_instances.scores.float() > 0.05]

        if len(pred_instances.scores) > 100:
            indices = pred_instances.scores.float().topk(max_num_boxes)[1]
            pred_instances = pred_instances[indices]
        output.pred_instances = pred_instances

        # predictions
        pred_instances = pred_instances.cpu().numpy()
        
        xyxy = pred_instances['bboxes']
        class_id = pred_instances['labels']
        confidence = pred_instances['scores']
        class_labels = []
        for iter in class_id:
            class_labels.append(texts[iter])
        return xyxy, class_labels, confidence

    def init_ram(self, cfg):
        """
        Init 2D Image Tagger RAM++
        """
        # load config
        transform = get_transform(image_size=384)
        # load model
        model = ram_plus(pretrained=cfg.foundation_model.ram_checkpoint, image_size=384, vit='swin_l')
        model.eval()
        device = 'cuda'
        model = model.to(device)
        print('------- Loaded RAM++ 384 SwinL -------')

        return transform, model

    def init_segmenter2d_models(self, cfg):
        """
        Init Segmenter 2D
        """
        # YOLO-World

        config = Config.fromfile(cfg.foundation_model.yoloworld_config_file)
        config.work_dir = "../temp" # temporary directory
        config.load_from = cfg.foundation_model.yoloworld_checkpoint
        runner = Runner.from_cfg(config)
        runner.call_hook("before_run")
        runner.load_or_resume()
        pipeline = config.test_dataloader.dataset.pipeline
        runner.pipeline = Compose(pipeline)

        # run model evaluation
        runner.model.eval()
        print('------- Loaded YOLO-World X -------')
        return runner

    def load_image(self, image_pil):
        """
        Grounding DINO preprocess
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image
