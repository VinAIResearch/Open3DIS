#!/bin/bash

dataset_cfg=${1:-'configs/scannetpp.yaml'}

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python3 tools/grounding_2d.py --config $dataset_cfg
    
#laion2b_s39b_b160k

