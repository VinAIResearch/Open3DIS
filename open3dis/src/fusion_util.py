import os
import torch
import glob
import math
import numpy as np
import nltk
import re
import torch_scatter


def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
                    intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

def NMS(bounding_boxes, confidence_score, label, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []
    # Bounding boxes
    boxes = np.array(bounding_boxes)
    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    # Confidence scores of bounding boxes
    score = np.array(confidence_score)
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_label = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    # Sort by confidence score of bounding boxes
    order = np.argsort(score)
    # Iterate bounding boxes
    # breakpoint()
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_label.append(label[index])
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, picked_label

def NMS_cuda(boxes, score, threshold):
    # If no bounding boxes, return empty list
    if len(boxes) == 0:
        return [], []
    # Bounding boxes
    # boxes = np.array(bounding_boxes)
    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]
    # Confidence scores of bounding boxes
    # score = np.array(confidence_score)
    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    # picked_label = []
    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    # Sort by confidence score of bounding boxes
    order = torch.argsort(score)
    # zero
    # Iterate bounding boxes
    # breakpoint()
    while len(order) > 0:
        # The index of largest confidence score
        index = order[-1]
        # Pick the bounding box with largest confidence score
        picked_boxes.append(boxes[index])
        picked_score.append(score[index])
        # picked_label.append(label[index])
        # Compute ordinates of intersection-over-union(IOU)
        x1 = torch.maximum(start_x[index], start_x[order[:-1]])
        x2 = torch.minimum(end_x[index], end_x[order[:-1]])
        y1 = torch.maximum(start_y[index], start_y[order[:-1]])
        y2 = torch.minimum(end_y[index], end_y[order[:-1]])
        # Compute areas of intersection-over-union
        w = (x2 - x1 + 1).clip(0.0)
        h = (y2 - y1 + 1).clip(0.0)
        # torch.maximum(0.0, x2 - x1 + 1)
        # h = torch.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        # left = np.where(ratio < threshold)
        order = order[:-1][(ratio < threshold)]

    return picked_boxes, picked_score

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def mask_nms(masks, scores, iou_threshold=0.5):
    # Sort masks based on scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    masks = masks[sorted_indices]
    scores = scores[sorted_indices]

    selected_indices = []

    while len(sorted_indices) > 0:
        current_mask = masks[0]
        current_score = scores[0]
        selected_indices.append(sorted_indices[0])

        sorted_indices = sorted_indices[1:]
        masks = masks[1:]
        scores = scores[1:]

        ious = [calculate_iou(current_mask, masks[i]) for i in range(len(masks))]
        ious = np.array(ious)

        overlapping_indices = np.where(ious > iou_threshold)[0]
        sorted_indices = np.delete(sorted_indices, overlapping_indices)
        masks = np.delete(masks, overlapping_indices, axis=0)
        scores = np.delete(scores, overlapping_indices)

    return selected_indices

def heuristic_nounex(caption, with_preposition):
    # NLP processing
    #nltk.set_proxy('http://proxytc.vingroup.net:9090/')
    #nltk.download("popular", quiet=True)
    #nltk.download("universal_tagset", quiet=True)
    if with_preposition:
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
                {<NBAR>} # If pattern is not found, just a single NBAR is ok
        """
    else:
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>} # If pattern is not found, just a single NBAR is ok
        """
    tokenized = nltk.word_tokenize(caption)
    chunker = nltk.RegexpParser(grammar)

    chunked = chunker.parse(nltk.pos_tag(tokenized))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    if current_chunk:
        named_entity = " ".join(current_chunk)
        if named_entity not in continuous_chunk:
            continuous_chunk.append(named_entity)

    return continuous_chunk

def get_nouns(caption):
    caption_words = []
    caption_words.extend(heuristic_nounex(caption, True))
    caption_words.extend(heuristic_nounex(caption, False))
    result = []
    for word in list(set(caption_words)):
        result.append(word.strip())
    
    tokenized_components = re.findall(r'\b\w+\b', caption)
    ordered_components = [component for component in result if component in tokenized_components]
    def custom_sort(item):
        return ordered_components.index(item)
    sorted_list = sorted(result, key=custom_sort)
    
    return sorted_list

def rotate_3d_feature_vector_anticlockwise_90(feature_vector):
    rotated_vector = feature_vector.permute(1, 0, 2)
    rotated_vector = torch.flip(rotated_vector, dims=(0,))

    return rotated_vector

def rotate_3db_feature_vector_anticlockwise_90(feature_vector):
    feature_vector = feature_vector.permute(0, 2, 3, 1)
    rotated_vector = feature_vector.permute(0, 2, 1, 3)
    rotated_vector = torch.flip(rotated_vector, dims=(1,))
    
    return rotated_vector.permute(0, 3, 1, 2)

def matrix_nms(proposals_pred, categories, scores, final_score_thresh=0.1, topk=-1):
    if len(categories) == 0:
        return proposals_pred, categories, scores

    ixs = torch.argsort(scores, descending=True)
    n_samples = len(ixs)

    categories_sorted = categories[ixs]
    proposals_pred_sorted = proposals_pred[ixs]
    scores_sorted = scores[ixs]

    # (nProposal, N), float, cuda
    intersection = torch.einsum(
        "nc,mc->nm", proposals_pred_sorted.type(scores.dtype), proposals_pred_sorted.type(scores.dtype)
    )
    proposals_pointnum = proposals_pred_sorted.sum(1)  # (nProposal), float, cuda
    ious = intersection / (proposals_pointnum[None, :] + proposals_pointnum[:, None] - intersection)

    # label_specific matrix.
    categories_x = categories_sorted[None, :].expand(n_samples, n_samples)
    label_matrix = (categories_x == categories_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (ious * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = ious * label_matrix

    # matrix nms
    sigma = 2.0
    decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
    compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
    decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = scores_sorted * decay_coefficient

    if topk != -1:
        _, get_idxs = torch.topk(
            cate_scores_update, k=min(topk, cate_scores_update.shape[0]), largest=True, sorted=False
        )
    else:
        get_idxs = torch.nonzero(cate_scores_update >= final_score_thresh).view(-1)

    return (
        proposals_pred_sorted[get_idxs],
        categories_sorted[get_idxs],
        cate_scores_update[get_idxs]
    )

def standard_nms(proposals_pred, categories, scores, threshold=0.2):
    ixs = torch.argsort(scores, descending=True)
    # n_samples = len(ixs)

    intersection = torch.einsum("nc,mc->nm", proposals_pred.type(scores.dtype), proposals_pred.type(scores.dtype))
    proposals_pointnum = proposals_pred.sum(1)  # (nProposal), float, cuda
    ious = intersection / (proposals_pointnum[None, :] + proposals_pointnum[:, None] - intersection)

    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)

        pivot_cls = categories[i]

        iou = ious[i, ixs[1:]]
        other_cls = categories[ixs[1:]]

        condition = (iou > threshold) & (other_cls == pivot_cls)
        # condition = (iou > threshold)
        remove_ixs = torch.nonzero(condition).view(-1) + 1

        remove_ixs = torch.cat([remove_ixs, torch.tensor([0], device=remove_ixs.device)]).long()

        mask = torch.ones_like(ixs, device=ixs.device, dtype=torch.bool)
        mask[remove_ixs] = False
        ixs = ixs[mask]
    get_idxs = torch.tensor(pick, dtype=torch.long, device=scores.device)

    return proposals_pred[get_idxs], categories[get_idxs], scores[get_idxs]