##!/bin/bash

export NCCL_P2P_DISABLE=1

LAUNCHER="accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --num_processes=3
    --num_machines=1
    --rdzv_conf rdzv_backend=static \
    --gpu_ids 0,1,2 \
    --main_process_port=10974 \
    /net/home/cmosig/projects/standing-deadwood/train.py \
    --config /net/home/cmosig/projects/standing-deadwood/config_mixvision_large_oversample_newdata.json \
"

$LAUNCHER
