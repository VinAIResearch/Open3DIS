#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python3 tools/grounding_2d.py --config configs/scannetpp.yaml
    
#laion2b_s39b_b160k

