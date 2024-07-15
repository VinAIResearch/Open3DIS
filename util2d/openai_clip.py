import torch
import numpy as np
import clip

class CLIP_OpenAI:
    def __init__(self, cfg):
        # CLIP
        self.clip_adapter, self.clip_preprocess = clip.load(cfg.foundation_model.clip_model, device = 'cuda')
        self.dim = 768
        print('------- Loaded CLIP ViT-L/14 336px OpenAI -------')