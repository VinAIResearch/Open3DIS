#!/usr/bin/env bash
GPUS=$2

OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=2 --master_port=$((RANDOM + 10000)) tools/train.py --dist configs/scannetpp/isbnet_scannetpp_benchmark.yaml --only_backbone --exp_name scannetpp_semantic