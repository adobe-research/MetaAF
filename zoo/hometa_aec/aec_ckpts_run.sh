#!/bin/bash
# uncomment --save_outputs to store the aec results

python hoaec_eval.py \
    --name aec_banded3 \
    --date 2022_05_23_00_04_11 --epoch 137 \
    --save_metrics --iwaenc_release # --save_outputs

python hoaec_eval.py \
    --name aec_banded9 \
    --date 2022_05_22_23_47_25 --epoch 126 \
    --save_metrics --iwaenc_release # --save_outputs

python hoaec_eval.py \
    --name aec_diag \
    --date 2022_05_23_00_01_35 --epoch 135 \
    --save_metrics --iwaenc_release # --save_outputs

