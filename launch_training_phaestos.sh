#!/bin/bash

export NCCL_P2P_DISABLE=1

EXPERIMENT="tversky_a04b06g2"

conda activate sd-env-12

LAUNCHER="accelerate launch \
    --multi_gpu \
    --mixed_precision=fp16 \
    --num_processes=3
    --num_machines=1
    --rdzv_conf rdzv_backend=static \
    --gpu_ids 1,2,3 \
    /net/home/jmoehring/standing-deadwood/train.py \
    --config /net/home/jmoehring/experiments/testing_runs/$EXPERIMENT/config.json
"

$LAUNCHER
