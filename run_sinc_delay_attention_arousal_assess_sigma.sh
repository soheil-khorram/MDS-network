#!/bin/sh
set -e

device=2
model="delay_attention"
data_provider="avec2016_provider"
export TF_CPP_MIN_LOG_LEVEL=3
exp_path="/z/Soheil/results/constraint_kernel_cnn/sinc_kernel_delay_attention_assess_sigma" 

for task in arousal
do
    for delay_num in 32
    do
        for channel_num in 16
        do
            for kernel_len in 8
            do
                for layer_num in 5
                do
                    for sigma in 0.000625 0.00125 0.0025 0.005 0.01 0.02 0.04 0.08 0.16
                    do
                        for seed in 1 2 3
                        do
                            this_exp_path=${exp_path}/${task}/sigma_${sigma}_delaynum_${delay_num}_channelnum_${channel_num}_kernellen_${kernel_len}_layernum_${layer_num}/seed_${seed}
                            echo "saving experiment to ${this_exp_path}"
                            CUDA_VISIBLE_DEVICES=${device} MODEL=${model} DATA_PROVIDER=${data_provider} python main.py\
                                --exp-dir ${this_exp_path}\
                                --task ${task}\
                                --seed ${seed}\
                                --conv-kernel-len ${kernel_len}\
                                --conv-channel-num ${channel_num}\
                                --conv-layer-num ${layer_num}\
                                --delay-num ${delay_num}\
                                --sigma ${sigma}\
                                --conv-l2-reg-weight 0.00\
                                --kernel-type sinc
                        done
                    done
                done
            done
        done
    done
done
