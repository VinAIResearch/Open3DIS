#!/bin/bash

dataset_cfg=${1:-'configs/scannetpp.yaml'}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python3 tools/generate_3d_inst.py --config $dataset_cfg