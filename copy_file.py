import os
import shutil
from tqdm import tqdm

root = "/home/tdngo/Workspace/3dis_ws/Open3DIS/data/s3dis/superpoints_notalign"
dst_root = "/home/tdngo/Workspace/3dis_ws/Open3DIS/data/s3dis/superpoints_open3dis"
files = os.listdir(root)

for file in tqdm(files):
    p = os.path.join(root, file )
    dst = os.path.join(dst_root, file)
    # dst = dst.replace('_inst_nostuff', '')
    shutil.copy(p, dst)