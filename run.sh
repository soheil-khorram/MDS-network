#!/bin/sh
set -e

device=2
export TF_CPP_MIN_LOG_LEVEL=3
model="delay_attention"
data_provider="avec2016_provider"
exp_path="/z/Soheil/results/constraint_kernel_cnn/tmp" 

CUDA_VISIBLE_DEVICES=${device} MODEL=${model} DATA_PROVIDER=${data_provider} python main.py\
    --exp-dir ${exp_path}\
    --task arousal\
    --nb-epochs 300\
    --lr 0.005\
    --conv-kernel-len 8\
    --conv-channel-num 16\
    --conv-layer-num 5\
    --sigma 0.005\
    --delay-num 16\
    --conv-l2-reg-weight 0.0\
    --kernel-type gaussian
