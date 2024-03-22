#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python3 tools/refine_grouding_feat.py --config configs/scannet200.yaml
    
#laion2b_s39b_b160k

