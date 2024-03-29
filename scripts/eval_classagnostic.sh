#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python3 open3dis/evaluation/eval_class_agnostic.py --config configs/scannetpp.yaml --type 2D
    
#laion2b_s39b_b160k

