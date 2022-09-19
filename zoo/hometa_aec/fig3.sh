#!/bin/bash

n_devices=2
batch_size=16
total_epochs=500
experiments_name="iwaenc_release_fig3"

h_sizes=( 16 32 64 )
group_sizes=( 3 5 9 17 33 )
group_modes=( "banded" "block" "diag")

for i in ${!group_modes[@]}; do

    group_mode=${group_modes[$i]}
     
    if [ ${group_mode} == "diag" ]; then
        
        group_size=1
        
        for k in ${!h_sizes[@]}; do
        
            h_size=${h_sizes[$k]}
            name=${experiments_name}_gm${group_mode}_gs${group_size}_hs${h_size}_te${total_epochs}
        
            echo "Training ${group_mode} aec model with hidden size of ${h_size}..."
            
            python hoaec.py \
                --n_frames 1 --window_size 4096 --hop_size 2048 --n_in_chan 1 --n_out_chan 1 \
                --is_real --n_devices ${n_devices} --batch_size ${batch_size} \
                --h_size ${h_size} --n_layers 2 --total_epochs ${total_epochs} \ 
                --val_period 1 --reduce_lr_patience 5 --early_stop_patience 16 \
                --name ${name} \
                --unroll 20 --double_talk --group_mode ${group_mode} \
                --group_size ${group_size} --ratio 1.0 --lr 0.0001 --random_roll
            
            echo "Evaluating ${group_mode} aec model with hidden size of ${h_size}..."
            
            date=$(ls -rt ./ckpts/${name} | tail -n 1) # assuming the last one
            epoch=$(basename $(ls ./ckpts/${name}/${date}/best* | tail -n 1))
            epoch="${epoch/best_ckpt_epoch_/""}" 
            epoch="${epoch/.pkl/""}"

            python hoaec_eval.py \
                --name ${name} \
                --date ${date} --epoch ${epoch} \
                --save_metrics # --save_outputs --generate_aec_data
            
        done
        
    else
    
        for j in ${!group_sizes[@]}; do
        
            group_size=${group_sizes[$j]}
            
            for k in ${!h_sizes[@]}; do
            
                h_size=${h_sizes[$k]}
                name=${experiments_name}_gm${group_mode}_gs${group_size}_hs${h_size}_te${total_epochs}
        
                echo "Training ${group_mode} aec model with group size of ${group_size} and hidden size of ${h_size}..."

                python hoaec.py \
                    --n_frames 1 --window_size 4096 --hop_size 2048 --n_in_chan 1 --n_out_chan 1 \
                    --is_real --n_devices ${n_devices} --batch_size ${batch_size} \
                    --h_size ${h_size} --n_layers 2 --total_epochs ${total_epochs} \
                    --val_period 1 --reduce_lr_patience 5 --early_stop_patience 16 \
                    --name ${name} \
                    --unroll 20 --double_talk --group_mode ${group_mode} \
                    --group_size ${group_size} --ratio 1.0 --lr 0.0001 --random_roll
                
                echo "Evaluating ${group_mode} aec model with group size of ${group_size} and hidden size of ${h_size}..."

                date=$(ls -rt ./ckpts/${name} | tail -n 1) # assuming the last one
                epoch=$(basename $(ls ./ckpts/${name}/${date}/best* | tail -n 1))
                epoch="${epoch/best_ckpt_epoch_/""}" 
                epoch="${epoch/.pkl/""}"

                python hoaec_eval.py \
                    --name ${name} \
                    --date ${date} --epoch ${epoch} \
                    --save_metrics # --save_outputs --generate_aec_data

            done
            
        done
        
    fi
    
done





