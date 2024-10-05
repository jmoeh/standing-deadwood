#!/bin/bash

export NCCL_P2P_DISABLE=1

EXPERIMENT="testing"

conda activate sd-env-12

LAUNCHER="accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    --num_processes=2
    --num_machines=1
    --rdzv_conf rdzv_backend=static \
    --gpu_ids 0,2,3 \
    /net/home/jmoehring/standing-deadwood/train.py \
    --config /net/home/jmoehring/experiments/$EXPERIMENT/config.json
"

$LAUNCHER
