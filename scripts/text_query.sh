#!/bin/bash

dataset_cfg=${1:-'configs/scannet200.yaml'}
text_query=${2:-'chair.table'}
clip_score=${3:-0.8}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python3 tools/text_query.py --config $dataset_cfg --text_query $text_query --clip_score $clip_score