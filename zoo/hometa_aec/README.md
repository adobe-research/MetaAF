<div align="center">

# Meta-Learning for Adaptive Filters with Higher-Order Frequency Dependencies

[Junkai Wu](https://www.linkedin.com/in/junkai-wu-19015b198/)<sup>1</sup>, [Jonah Casebeer](https://jmcasebeer.github.io)<sup>1</sup>, [Nicholas J. Bryan](https://ccrma.stanford.edu/~njb/)<sup>2</sup>, and [Paris Smaragdis](https://paris.cs.illinois.edu/)<sup>1</sup>

<sup>1</sup> Department of Computer Science, University of Illinois at Urbana-Champaign<br>
<sup>2</sup> Adobe Research<br>
</div>

 <!-- START doctoc generated TOC please keep comment here to allow auto update -->
 <!-- doctoc --maxlevel 2 README.md -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Meta-Learning for Adaptive Filters with Higher-Order Frequency Dependencies](#meta-learning-for-adaptive-filters-with-higher-order-frequency-dependencies)
  - [Abstract](#abstract)
  - [Demos](#demos)
  - [Code](#code)
  - [Data Download](#data-download)
  - [Pretrained Checkpoints](#pretrained-checkpoints)
  - [License](#license)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Abstract

Adaptive filters are applicable to many signal processing tasks including acoustic echo cancellation, beamforming, and more. Adaptive filters are typically controlled using algorithms such as least-mean squares(LMS), recursive least squares(RLS), or Kalman filter updates. Such models are often applied in the frequency domain, assume frequency independent processing, and do not exploit higher-order frequency dependencies, for simplicity. Recent work on meta-adaptive filters, however, has shown that we can control filter adaptation using neural networks without manual derivation, motivating new work to exploit such information. In this work, we present higher-order meta-adaptive filters, a key improvement to meta-adaptive filters that incorporates higher-order frequency dependencies. We demonstrate our approach on acoustic echo cancellation and develop a family of filters that yield multi-dB improvements over competitive baselines, and are at least an order-of-magnitude less complex. Moreover, we show our improvements hold with or without a downstream speech enhancer.

For more details, please see:
"[Meta-Learning for Adaptive Filters with Higher-Order Frequency Dependencies](https://arxiv.org/abs/2209.09955)", [Junkai Wu](https://www.linkedin.com/in/junkai-wu-19015b198/), [Jonah Casebeer](https://jmcasebeer.github.io), [Nicholas J. Bryan](https://ccrma.stanford.edu/~njb/), and [Paris Smaragdis](https://paris.cs.illinois.edu/), IWAENC, 2022. If you use ideas or code from this work, please cite our paper:

```BibTex
@article{wu2022metalearning,
  title={Meta-Learning for Adaptive Filters with Higher-Order Frequency Dependencies},
  author={Wu, Junkai and Casebeer, Jonah and Bryan, Nicholas J. and Smaragdis, Paris},    
  journal={arXiv preprint arXiv:2209.09955},
  year={2022},
}
```

## Demo

For audio demonstrations of this work, please check out our [demo website](https://jmcasebeer.github.io/metaaf/higher-order). You'll be able to find demos for AEC and as well as joint AEC and speech enhancement.

## Code

This folder contains the implementation of our paper "[Meta-Learning for Adaptive Filters with Higher-Order Frequency Dependencies](https://arxiv.org/abs/2209.09955)". It leverages the `metaaf` python package. This directory contains all necessary code to reproduce and run our experiments. The core file is `hoaec.py`. It contains the main AEC code and training configuration. The `hoaec_joint.py` file contains code for joint AEC and speech enhancement.

### Train and Evaluate Higher Order Meta-AEC

Run this command to train Meta-AEC model with `banded` dependency structure, `group_size` of 3, and `hidden_size` of 32. Change the arguments to train other models.

```{bash}
python hoaec.py \
    --n_frames 1 --window_size 4096 --hop_size 2048 --n_in_chan 1 --n_out_chan 1 \
    --is_real --n_devices 1 --batch_size 16 \
    --h_size 32 --n_layers 2 --total_epochs 500 \
    --val_period 1 --reduce_lr_patience 5 --early_stop_patience 16 \
    --name aec_32_banded3 \
    --unroll 20 --double_talk --group_mode banded \
    --group_size 3 --lr 0.0001 --random_roll
```

Replace the `date` and `epoch` arguments and use this command to evaluate the trained Meta-AEC model.

```{bash}
python hoaec_eval.py \
    --name aec_32_banded3 \
    --date <date> --epoch <epoch> \
    --save_metrics --save_outputs
```

You can also run `/hometa_aec/fig3.sh` to train and evaluate models shown in fig. 3 of the paper all at once. To calculate `SERLE` we used in the paper, check out `/hometa_aec/aec_results.ipynb`.

### Train and Evaluate DNN Speech Enhancer

Replace the `date` and `epoch` arguments and use this command to store the `aec` outputs of the trained Meta-AEC model.

```{bash}
python hoaec_eval.py \
    --name aec_32_banded --date <aec_date> --epoch <aec_epoch> \
    --generate_aec_data --fix_train_roll
```

Then run this command to train the DNN speech enhancer.

```{bash}
python hoaec_joint.py \
    --window_size 4096 --hop_size 256 --is_real 
    --n_devices 1 --batch_size 32 \
    --h_size 32 --n_layers 2 
    --total_epochs 200 \
    --val_period 1 --reduce_lr_patience 5 --early_stop_patience 16 \
    --unroll 150 --double_talk \
    --group_mode block --group_size 4 \
    --lr 0.0006 \
    --m_n_frames 1 --m_window_size 512 --m_hop_size 256 \
    --m_n_in_chan 1 --m_n_out_chan 1 --m_is_real \
    --joint_mode aec_res --aec_res_mode train\
    --name res_32_banded --aec_name aec_32_banded
```

Replace the `date` and `epoch` arguments and run this command to evaluate the trained DNN-RES model.

```{bash}
python hoaec_joint_eval.py \
    --mode res --name res_32_banded --date <date> --epoch <epoch> \
    --aec_name aec_32_banded --save_metrics --save_outputs
```

## Data Download

Follow the instructions [here](https://github.com/microsoft/AEC-Challenge) to get and download the Microsoft acoustic echo cancellation challenge dataset. Unzip the dataset and set the base path for `AEC_DATA_DIR` in `/zoo/__config__.py`. Also set `RES_DATA_DIR` in `/zoo/__config__.py` to a directory you want to store the `aec`
outputs for `res` training and evaluation.

## Pretrained Checkpoints

### Download Checkpoints

You can download the checkpoints for Meta-AEC models and DNN speech enhancers in fig. 4 of the paper with [this link](https://drive.google.com/file/d/1eA1_ZMVLPkeOolM3Zf2H2JE2mAgM2lsH/view?usp=sharing). Unzip and make sure the folders `aec_diag`, `aec_banded9`, `aec_banded3`, `res_diag`, `res_banded9` and `res_banded3` are under the `/hometa_aec/ckpts` folder.

### Run Checkpoints

You can use `/hometa_aec/aec_ckpts_run.sh` to run the released `aec` models and `/hometa_aec/res_ckpts_run.sh` to run the released `res` models. Please keep the `--iwaenc_release` flag on to use the correct filter. In the released code we added anti-aliasing for gradient and error terms, which the released models were not trained with.

## License

All core utility code within the `metaaf` folder is licensed via the [University of Illinois Open Source License](../metaaf/LICENSE). All code within the `zoo` and `hometa_aec` folders and model weights are licensed via the [Adobe Research License](LICENSE). Copyright (c) Adobe Systems Incorporated. All rights reserved.
