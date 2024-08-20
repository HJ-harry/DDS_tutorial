#!/bin/bash

declare -a Nview_list=(8)
declare -a T_sampling_list=(50)
declare -a eta_list=(0.85)

for Nview in "${Nview_list[@]}"; do
    echo $Nview
    for ((i=0; i<=1; i++))
    do
        T_sampling=${T_sampling_list[i]}
        eta=${eta_list[i]}
        echo $T_sampling
        echo $eta
        python main.py \
        --type 2d \
        --config AAPM256.yml \
        --Nview $Nview \
        --eta $eta \
        --deg "SV-CT" \
        --sigma_y 0.0 \
        --T_sampling 100 \
        --rho 10.0 \
        --lamb 0.04 \
        --T_sampling $T_sampling \
        -i ./results_3d
    done
done