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
#### Grounding DINO
from detectron2.structures import BitMasks
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
#### RAM Plus
from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
#### Util
from util2d.util import show_mask, masks_to_rle 
#### Open3DIS util
from open3dis.dataset import build_dataset
from open3dis.dataset.scannet_loader import ScanNetReader, scaling_mapping
from open3dis.src.fusion_util import NMS_cuda
from open3dis.src.mapper import PointCloudToImageMapper

class RAM_Grounded_Sam:
    ###################################################################
    #                     RAM Grounding DINO + SAM                    #
    ###################################################################
    def __init__(self, cfg):
        # Load Foundation Model
        sam2d = SAM_HQ(cfg)
        clip2d = CLIP_OpenAI(cfg)
        self.sam_predictor = sam2d.sam_predictor
        self.clip_adapter, self.clip_preprocess = clip2d.clip_adapter, clip2d.clip_preprocess
        # Load RAM++
        self.ram_transform, self.ram_model = self.init_ram(cfg)
        # Load Grounding DINO 2D
        self.grounding_dino_model = self.init_segmenter2d_models(cfg)

    def gen_grounded_mask_and_feat(self, scene_id, class_names, cfg, gen_feat=True):
        """
        RAM + Grounding DINO + SAM, CLIP
            Generate 2D masks from GDino RAM output
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
            #### Processing RAM++ ####
            image = self.ram_transform(image_pil).unsqueeze(0).to('cuda')
            res = inference(image, self.ram_model)
            res = res[0].split('|')
            res = [st.strip() for st in res]

            ### RAM querying into multiple chunks -- see Supplementary
            segment_size = 2 # scannetpp_benchmark
            segments = [res[i : i + segment_size] for i in range(0, len(res), segment_size)]
            for cls_name in segments:
                boxes, confidences = self.get_grounding_output(
                    image_infer,
                    ".".join(cls_name),
                    cfg.foundation_model.box_threshold,
                    cfg.foundation_model.text_threshold,
                    device=cfg.foundation_model.device,
                )

                if len(boxes) > 0:
                    boxes_filt.append(boxes)
                    confs_filt.append(confidences)
            if len(boxes_filt) == 0:  # No box in that view
                continue
            boxes_filt = torch.cat(boxes_filt)
            confs_filt = torch.cat(confs_filt)

            size = image_pil.size
            H, W = size[1], size[0]
            boxes_filt = boxes_filt * torch.Tensor([W, H, W, H])[None, ...].cuda()

            # XYWH to XYXY
            boxes_filt[:, :2] -= boxes_filt[:, 2:] / 2
            boxes_filt[:, 2:] += boxes_filt[:, :2]
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

            # BOX NMS
            boxes_filt, confs_filt = NMS_cuda(boxes_filt, confs_filt, 0.5) # -> number of box dec
            boxes_filt = torch.stack(boxes_filt)
            confs_filt = torch.tensor(confs_filt)

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
                os.makedirs("../debug/scannetpp/" + scene_id, exist_ok=True)
                plt.savefig(
                    os.path.join("../debug/scannetpp/" + scene_id + "/sam_" + str(i) + ".jpg"),
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=0.0,
                )

            #### SAVING MASKS, CLIP FEATURES
            grounded_data_dict[frame_id] = {
                "masks": masks_to_rle(masks),
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

    def get_grounding_output(self, image, caption, box_threshold, text_threshold, with_logits=True, device="cuda"):
        """
        RAM Grounding DINO box generator
        Returning boxes and logits scores for each chunk in the caption with box & text threshoding
        """

        # Caption formatting
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."

        self.grounding_dino_model = self.grounding_dino_model.to(device)
        image = image.to(device)

        # Grounding DINO box generator
        with torch.no_grad():
            outputs = self.grounding_dino_model(image[None], captions=[caption])
            logits = outputs["pred_logits"].sigmoid()[0]  # (nqueries, 256)
        boxes = outputs["pred_boxes"][0]  # (nqueries, 4)

        # Filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        return boxes_filt, logits_filt.max(dim=1)[0]

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
        # Grounding DINO
        grounding_dino_model = self.load_model(
            cfg.foundation_model.grounded_config_file, cfg.foundation_model.grounded_checkpoint, device="cuda")
        print('------- Loaded Grounding DINO OGC SwinT -------')
        return grounding_dino_model

    def load_model(self, model_config_path, model_checkpoint_path, device):
        """
        Grounding DINO loader
        """
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model.cuda()
        return model

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
