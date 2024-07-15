#!/bin/bash

dataset_cfg=${1:-'configs/scannet200.yaml'}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python3 tools/reproduce_openyolo3d.py --config $dataset_cfg