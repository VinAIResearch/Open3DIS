import argparse
import csv
import os

import numpy as np
import open_clip
import torch
import util
import util_3d


CLASS_LABELS = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
]
VALID_ID = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
UNKNOWN_ID = np.max(VALID_ID) + 1

#### Scenes
scenes = []
with open("../../Dataset/ScannetV2/queries_val_scannet.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        scenes.append(row)
scenes.pop(0)
######


def get_parser():
    parser = argparse.ArgumentParser("Evaluation ScannetV2", add_help=True)
    ### Features
    parser.add_argument("--exp", type=str, required=True, help="experiment workspace")
    parser.add_argument("--feature_stage", type=str, required=True, help="feature type")
    parser.add_argument("--predsem_path", type=str, required=True, help="pred sem path")
    parser.add_argument("--gtsem_path", type=str, required=True, help="gt path pointcloud")
    parser.add_argument("--positive_thresh", type=float, required=True, help="semantic positive threshold")

    return parser


# Semantic Evaluation not include other furniture
class Scannetv2SemEval(object):
    def __init__(
        self,
        num_class=19,
        ignore_label=-100,
        predsem_path=None,
        feature=None,
        exp=None,
        gtsem_path=None,
        pos_thres=None,
    ):
        self.ignore_label = ignore_label
        self.num_classes = num_class
        self.predsem = predsem_path
        self.feature = feature  # Feature stage
        self.gtsem = gtsem_path  # pcl path
        self.exp = exp
        self.adapter, _, _ = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
        self.pos_thres = pos_thres

    def convert_sem_result(self, outputpath):

        for scene_idd in scenes:
            scene_id = scene_idd[0]
            gt_file = os.path.join(self.gtsem, scene_id + "_inst_nostuff.pth")
            gt_scene = torch.load(gt_file)[2]  # Semantic
            result = torch.zeros((gt_scene.shape[0])) + self.ignore_label  # -100 ignore ids
            confusion_table = torch.zeros((gt_scene.shape[0], self.num_classes))

            subfolder = os.listdir(self.predsem)
            sem_res = [name for name in subfolder if name.startswith(self.exp)]
            for class_folder in sem_res:
                path = os.path.join(self.predsem, class_folder)
                scene_path = os.path.join(path, os.path.join(self.feature, scene_id + "_grounded_ov.pt"))
                class_name = class_folder.replace(self.exp + "_", "").replace("_", " ")
                print(class_name)
                index = [iter for iter in range(len(CLASS_LABELS)) if CLASS_LABELS[iter] == class_name][0]
                point_features = torch.load(scene_path)["feat"]
                with torch.no_grad(), torch.cuda.amp.autocast():
                    self.adapter.cuda()
                    txts = [class_name, "other"]
                    text = open_clip.tokenize(txts)
                    text_features = self.adapter.encode_text(text.cuda()).cuda()
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    confusion_table[:, index] = (50.0 * point_features.half().cuda() @ text_features.T).softmax(
                        dim=-1
                    )[:, 0]
            temp = torch.max(confusion_table, dim=-1)
            predicted_value, predicted_class = temp[0], temp[1]
            positive_ind = torch.where(predicted_value > self.pos_thres)[0]  # positive points
            result = result.to(torch.int16)
            predicted_class = predicted_class.to(torch.int16)
            result[positive_ind] = predicted_class[positive_ind]
            torch.save(result, os.path.join(outputpath, scene_id + ".pth"))
            breakpoint()

    def evaluate_scan(self, pred_file, gt_file, confusion):
        pred_file = torch.load()
        for gt_val, pred_val in izip(gt_ids.flatten(), pred_ids.flatten()):
            if gt_val not in VALID_ID:
                continue
            if pred_val not in VALID_ID:
                pred_val = UNKNOWN_ID
            confusion[gt_val][pred_val] += 1

    def get_iou(self, label_id, confusion):
        if not label_id in VALID_ID:
            return float("nan")
        # #true positives
        tp = np.longlong(confusion[label_id, label_id])
        # #false negatives
        fn = np.longlong(confusion[label_id, :].sum()) - tp
        # #false positives
        not_ignored = [l for l in VALID_ID if not l == label_id]
        fp = np.longlong(confusion[not_ignored, label_id].sum())

        denom = tp + fp + fn
        if denom == 0:
            return float("nan")
        return (float(tp) / denom, tp, denom)

    def write_result_file(self, confusion, ious, filename):
        with open(filename, "w") as f:
            f.write("iou scores\n")
            for i in range(len(VALID_ID)):
                label_id = VALID_ID[i]
                label_name = CLASS_LABELS[i]
                iou = ious[label_name][0]
                f.write("{0:<14s}({1:<2d}): {2:>5.3f}\n".format(label_name, label_id, iou))
            f.write("\nconfusion matrix\n")
            f.write("\t\t\t")
            for i in range(len(VALID_ID)):
                # f.write('\t{0:<14s}({1:<2d})'.format(CLASS_LABELS[i], VALID_ID[i]))
                f.write("{0:<8d}".format(VALID_ID[i]))
            f.write("\n")
            for r in range(len(VALID_ID)):
                f.write("{0:<14s}({1:<2d})".format(CLASS_LABELS[r], VALID_ID[r]))
                for c in range(len(VALID_ID)):
                    f.write("\t{0:>5.3f}".format(confusion[VALID_ID[r], VALID_ID[c]]))
                f.write("\n")
        print("wrote results to", filename)

    def evaluate(self, pred_files, gt_files, output_file):
        max_id = UNKNOWN_ID
        confusion = np.zeros((max_id + 1, max_id + 1), dtype=np.ulonglong)

        print("evaluating", len(pred_files), "scans...")
        for i in range(len(pred_files)):
            evaluate_scan(pred_files[i], gt_files[i], confusion)
            sys.stdout.write("\rscans processed: {}".format(i + 1))
            sys.stdout.flush()
        print("")

        class_ious = {}
        for i in range(len(VALID_ID)):
            label_name = CLASS_LABELS[i]
            label_id = VALID_ID[i]
            class_ious[label_name] = get_iou(label_id, confusion)
        # print
        print("classes          IoU")
        print("----------------------------")
        for i in range(len(VALID_ID)):
            label_name = CLASS_LABELS[i]
            # print('{{0:<14s}: 1:>5.3f}'.format(label_name, class_ious[label_name][0]))
            print(
                "{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})".format(
                    label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]
                )
            )
        write_result_file(confusion, class_ious, output_file)


if __name__ == "__main__":
    args = get_parser().parse_args()

    sempth = os.path.join(args.predsem_path, "final_sem")
    try:
        os.makedirs(sempth)
        print("Created directory !")
    except:
        print("Directory existed")
    eval = Scannetv2SemEval(
        num_class=19,
        ignore_label=-100,
        predsem_path=args.predsem_path,
        feature=args.feature_stage,
        exp=args.exp,
        gtsem_path=args.gtsem_path,
        pos_thres=args.positive_thresh,
    )
    eval.convert_sem_result(sempth)
