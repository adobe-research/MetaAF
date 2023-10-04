<div align="center">

# Scaling Up Adaptive Filter Optimizers

[Jonah Casebeer](https://jmcasebeer.github.io)<sup>1</sup>, [Nicholas J. Bryan](https://ccrma.stanford.edu/~njb/)<sup>2</sup>, and [Paris Smaragdis](https://paris.cs.illinois.edu/)<sup>1</sup>

<sup>1</sup> Department of Computer Science, University of Illinois at Urbana-Champaign<br>
<sup>2</sup> Adobe Research<br>
</div>

 <!-- START doctoc generated TOC please keep comment here to allow auto update -->
 <!-- doctoc --maxlevel 2 README.md -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Scaling Up Adaptive Filter Optimizers](#scaling-up-adaptive-filter-optimizers)
  - [Abstract](#abstract)
  - [Demo](#demo)
  - [Code](#code)
  - [Data Download](#data-download)
  - [Pretrained Checkpoints](#pretrained-checkpoints)
  - [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Abstract

We introduce a new online adaptive filtering method called supervised multi-step adaptive filters (SMS-AF). Our method uses neural networks to control or optimize linear multi-delay or multi-channel frequency-domain filters and can flexibly scale-up performance at the cost of increased compute --- a property rarely addressed in the AF literature, but critical for many applications. To do so, we extend recent work with a series of improvements including feature pruning, a supervised loss, and multiple optimization steps per time-frame. These improvements work together in a cohesive manner to unlock scaling. Furthermore, we show how our method relates to Kalman filtering and meta-adaptive filtering, making it seamlessly applicable to a diverse set of AF tasks. We evaluate our method on acoustic echo cancellation (AEC) and multi-channel speech enhancement tasks and compare against several baselines on standard synthetic and real-world datasets. Results show our method performance scales with inference cost and model capacity, yields multi-dB performance gains for both tasks, and is real-time capable on a single CPU core.

For more details, stay tuned -- the paper is under review.

## Demo

Demos are available [here](https://jmcasebeer.github.io/metaaf/sms-af).

## Code

This folder contains the implementation of our paper. It uses the `metaaf` python package. This directory contains all necessary code to reproduce and run our experiments. The core files are `aec.py` and `gsc.py`, they contain the code to train an SMS-AF for AEC and GSC tasks. The `aec_eval.py` and `gsc_eval.py` files contain code to evaluate trained models. The other files contain various utilities and baselines modules.

### Train and Evaluate

We release checkpoints for all models described in this paper. If you wish to train your own, instructions below.

Run this command to train a PU supevised AEC model. You can find more example and details in the `aec.py` file.

```{bash}
python aec.py --n_frames 8 --window_size 512 --hop_size 256 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 16 --group_size 5 --group_hop 2 --h_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 3 --unroll 128 --name sms_aec_m_s_pu --features_to_use uef --no_inner_grad --loss sup_echo_td_ola --val_loss neg_sisdr_ola --inner_iterations 1 --auto_posterior --no_analysis_window
```

For evaluation, replace the `name`, `date` and `epoch` arguments in the below command. Append `--real_data` to test on the real dataset.

```{bash}
python aec_eval.py --name <checkpoint name> --date <checkpoint date>  --epoch <checkpoint epoch> --save_metrics
```

Run this command to train a PUPU supevised GSC model.

```{bash}
python gsc.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --n_devices 1 --batch_size 16 --group_size 5 --group_hop 2 --h_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 3 --name gsc_sisdr_val_sisdr_2iter_posterior --unroll 128 --cov_init identity --cov_update oracle --exp_avg .9 --steer_method rank1 --cov_update_regularizer 0.01 --loss sisdr --val_metric neg_sisdr --lr 5e-4 --inner_iterations 2 --auto_posterior --features_to_use uef
```

Evaluation is just like AEC, replace the `name`, `date` and `epoch` arguments in the below command.

```{bash}
python gsc_eval.py --name <checkpoint name> --date <checkpoint date>  --epoch <checkpoint epoch> --save_metrics
```

## Data Download

Follow the instructions [here](https://github.com/microsoft/AEC-Challenge) to get and download the Microsoft acoustic echo cancellation challenge dataset. Follow the instructions [here](https://catalog.ldc.upenn.edu/LDC2017S24) to get and download the CHIME3 challenge dataset.

## Pretrained Checkpoints

We have released all SMS-AF checkpoints in a tagged draft release. You should unzip and move all the checkpoints into `/sms_af/ckpts`.

## License

All core utility code within the `metaaf` folder is licensed via the [University of Illinois Open Source License](../metaaf/LICENSE). All code within the `zoo` and `sms_af` folders and model weights are licensed via the [Adobe Research License](LICENSE). Copyright (c) Adobe Systems Incorporated. All rights reserved.
