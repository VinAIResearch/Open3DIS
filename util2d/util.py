import torch
import numpy as np
import pycocotools.mask
from typing import Dict, Union

def show_mask(mask, ax, random_color=False):
    """
    Mask visualization
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def masks_to_rle(masks) -> Dict:
    """
    Encode 2D mask to RLE (save memory and fast)
    """
    res = []
    if masks == None:
        return None
    masks = masks.squeeze(1)
    for mask in masks:
        if torch.is_tensor(mask):
            mask = mask.detach().cpu().numpy()
        assert isinstance(mask, np.ndarray)
        rle = pycocotools.mask.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        res.append(rle)
    return res

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
