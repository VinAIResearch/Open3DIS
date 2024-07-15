#!/bin/bash

export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH

export PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python3 tools/maskwise_vocab.py --config configs/scannetpp.yaml --type 2D
    
#laion2b_s39b_b160k

