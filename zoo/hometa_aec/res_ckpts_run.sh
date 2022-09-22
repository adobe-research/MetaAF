#!/bin/bash
# uncomment --save_outputs to store the aec results

aec_name="aec_banded3"
aec_date="2022_05_23_00_04_11"
aec_epoch=137

res_name="res_banded3"
res_date="2022_05_24_23_06_59"
res_epoch=119

# store aec outputs
python hoaec_eval.py \
    --name ${aec_name} --date ${aec_date} \
    --epoch ${aec_epoch} --generate_aec_data \
    --fix_train_roll --iwaenc_release
    
# eval res
python hoaec_joint_eval.py \
    --mode res --name ${res_name} --date ${res_date} --epoch ${res_epoch} \
    --aec_name ${aec_name} --save_metrics --save_outputs





