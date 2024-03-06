import torch
import os
from tqdm import tqdm

files = os.listdir("exp/version_test/grounded_feat")

for file in tqdm(files):
    data = torch.load(os.path.join("exp/version_test/grounded_feat", file))['feat']

    torch.save({
        'feat': data.half()
    },
    os.path.join("exp/version_test/grounded_feat", file)
    )