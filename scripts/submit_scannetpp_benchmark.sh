#!/bin/bash

dataset_cfg=${1:-'configs/scannetpp_benchmark_instance_test.yaml'}
export PYTHONWARNINGS="ignore"
PYTHONPATH=./:$PYTHONPATH
export PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python3 tools/submit_scannetpp_benchmark.py --config $dataset_cfg