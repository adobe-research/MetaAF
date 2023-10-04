<div align="center">

# Meta-AF Echo Cancellation for Improved Keyword Spotting

[Jonah Casebeer](https://jmcasebeer.github.io)<sup>1</sup>, [Junkai Wu](https://www.linkedin.com/in/junkai-wu-19015b198/)<sup>1</sup>, and [Paris Smaragdis](https://paris.cs.illinois.edu/)<sup>1</sup>

<sup>1</sup> Department of Computer Science, University of Illinois at Urbana-Champaign<br>
</div>

 <!-- START doctoc generated TOC please keep comment here to allow auto update -->
 <!-- doctoc --maxlevel 2 README.md -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Meta-AF Echo Cancellation for Improved Keyword Spotting](#meta-af-echo-cancellation-for-improved-keyword-spotting)
  - [Abstract](#abstract)
  - [Code](#code)
  - [Data Download](#data-download)
  - [Pretrained Checkpoints](#pretrained-checkpoints)
  - [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Abstract

Adaptive filters (AFs) are vital for enhancing the performance of downstream tasks, such as speech recognition, sound event detection, and keyword spotting. However, traditional AF design prioritizes isolated signal-level objectives, often overlooking downstream task performance due to the labor-intensive manual tuning process. This can lead to suboptimal performance. Recent research has leveraged meta-learning to automatically learn AF update rules from data, alleviating the need for manual tuning when using simple signal-level objectives. This paper improves the Meta-AF framework by expanding it to support end-to-end training for arbitrary downstream tasks. We focus on classification tasks, where we introduce a novel training methodology that harnesses self-supervision and classifier feedback. We evaluate our approach on the combined task of acoustic echo cancellation and keyword spotting. Our findings demonstrate consistent performance improvements with both pre-trained and joint-trained keyword spotting models across synthetic and real playback. Notably, these improvements come without requiring additional tuning, increased inference-time complexity, or reliance on oracle signal-level training data.

For more details, stay tuned -- the paper is under review.

## Code

This folder contains the implementation of our paper. It uses the `metaaf` python package. This directory contains all necessary code to reproduce and run our experiments. The core file is `ct_meta_aec.py`. It contains the main AEC code and training configuration for classification training. The `train_kws.py` file contains code for pre-training KWS modules, and the other files contain various utilities and baselines modules.

### Train and Evaluate

We release checkpoints for all models described in this paper. If you wish to train your own, instructions below.

Run this command to train a KWS, and change the value of `--n_labels` to train models for the 35, 10, and 2 keyword setups. You can find example commands for training the baseline models in the `train_kws.py` file.

```{bash}
python train_kws.py --name 35cmds_tcn --n_labels 35 --block_h 128 --n_blocks 3 --n_mel 40 --kws_window_size 512 --kws_hop_size 256
```

Run this command to perform classification training with a pre-trained KWS.

```{bash}
python ct_metaaec_kws.py --n_frames 4 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 16 --group_size 5 --group_hop 2 --h_size 48 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 3 --name example_ct_metaaec --unroll 93 --kws_loc ./ckpts/35cmds_tcn.pkl --kws_mode 35cmds --lr 2e-4 --outer_loss_alpha 0.5 --debug
```

Run this command to perform joint classification training with a pre-trained KWS and AEC. Update the `--aec_init_ckpt` values to match whatever checkpoint you produced with the previous training.

```{bash}
python ct_metaaec_kws.py --n_frames 4 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 16 --group_size 5 --group_hop 2 --h_size 48 --total_epochs 1000 --val_period 5 --reduce_lr_patience 1 --early_stop_patience 5 --name example_jct_metaaec --unroll 93 --kws_loc ./ckpts/35cmds_tcn.pkl --kws_mode 35cmds --outer_loss_alpha 0.50 --joint_train_kws --use_kws_init_ckpt --use_aec_init_ckpt --aec_init_ckpt 5_kws_loss_35cmds_med_25_0 2023_03_27_13_42_45 300 --lr 1e-4 --max_norm 1 --b1 0.9
```

For evaluation, replace the `name`, `date` and `epoch` arguments in the below command. Append `--real_data` to test on the real dataset and chage the kws arguments to exaluate different keyword model setups.

```{bash}
python ct_metaaec_eval.py --name 5_kws_loss_35cmds_med_25_0 --date 2023_03_27_13_42_45 --epoch 300 --kws_loc ./ckpts/35cmds_tcn.pkl --kws_mode 35cmds --save_metrics
```

## Data Download

Follow the instructions [here](https://github.com/microsoft/AEC-Challenge) to get and download the Microsoft acoustic echo cancellation challenge dataset. Use the link [here](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) to get a zip of the google speech commands dataset (v2).

## Pretrained Checkpoints

We have released all kws, ct-meta-aec, and baseline model checkpoints in a tagged draft release. You should unzip and move all the checkpoints into `/ct_af/ckpts`.

## License

All core utility code within the `metaaf` folder is licensed via the [University of Illinois Open Source License](../metaaf/LICENSE). All code within the `zoo` and `ct_af` folders and model weights are licensed via the [Adobe Research License](LICENSE). Copyright (c) Adobe Systems Incorporated. All rights reserved.
