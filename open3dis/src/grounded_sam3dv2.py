
import numpy as np
import torch
import matplotlib.pyplot as plt

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
##############################################


def load_image(image_pil):

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            # T.RandomResize([400], max_size=400),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model.cuda()


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

        
    
    # probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
    logits = outputs["pred_logits"].sigmoid()[0]  # (nqueries, 256)
    boxes = outputs["pred_boxes"][0]  # (nqueries, 4)
    

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    # logits_filt.shape[0]

    return boxes_filt, logits_filt.max(dim=1)[0]




    

