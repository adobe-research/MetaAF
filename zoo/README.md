# Meta-AF Zoo

The Meta-AF Zoo folder contains implementations of system identification, acoustic echo cancellation, equalization, weighted predection error dereverberation, and a generalized sidelobe canceller beamformer as developed in our paper.


The zoo has sub-directories for each of these tasks: `aec`, `eq`, `wpe`, and `gsc`. Note the `sysid` folder redirects to the `aec` within near-end signal set to zero. We walk through an example usage of the `aec` modules below. All modules share the same general structure. Though, a key difference is that the `aec` and `eq` modules use the `optimizer_gru` and  `wpe`/`gsc` use `optimizer_fgru`. These two optimizer variants are identical for single-frame/channel filters but differ in how they scale past that. The fgru couple frames/channels and is significantly faster and performs better.

[**Zoo Structure**](#zoo-structure)
| [**Run A Model**](#running-a-zoo-model)
| [**Install guide**](#installation)
| [**Data Download**](#data-download)

## Zoo Structure

The `aec` task and sub-directory has `aec.py`, `aec_baselines.py`, and `aec_eval.py` files. The first constains the filter defitions, dataset, and is the entry point for training a model. The second file is used to tune baselines and produces a tuned baseline checkpoint. The third file is used for evaluation and uses checkpoints produced from the first or second files. As seen below, all tasks have these same three files. The `aec` module contains two extra baselines, a Kalman filter, and Speex whereas the `wpe` module only has the NARA-WPE baseline. The additional `metrics.py` contains implementations or wrappers of common metrics such as SNR, SRR, SI-SDR, STOI, and their segmental variations. 

## Running a Zoo Model

To train a model, navigate to the task directory and run `<task>.py` with the desired arguments. You can find the commands we used to run all our experiments in each `<task>.py` file. For example, to train an AEC model with a 2048 window, 1024 hop, on the nonlinear dataset with double-talk all on a single-gpu with batch size 64 do:

```{bash}
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_aec_16_dt_nl --unroll 16 --extra_signals ude --random_roll --outer_loss log_self_mse --double_talk --dataset nonlinear
```

You can then run eval on that model and save the metrics and outputs by running

```{bash}
python aec_eval.py --name meta_aec_16_dt_nl --date 2022_03_17_11_23_22 --epoch 200 --save_metrics --save_outputs
```

where you replace `name/date/epoch` with the desired values. If you use one of the multichannel models, make sure to add `--model_type fgru`. If you want to run a baseline model, make sure to specify the model type via `--model_type <lms/rms/nlms/rls>`.

To run a baseline model, you need to have a checkpoint. You can make one by running baseline tuning. In the case of AEC and an RLS optimizer, you can produced a tuned checkpoint in the same format as the Meta-AEC model via:

```{bash}
python aec_baselines.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --batch_size 32 --total_epochs 0 --n_devices 1 --name aec_dt_nl_rls --optimizer rls --optimize_conjugate --random_roll --double_talk --dataset nonlinear
```

## Installation

To run any of the models in the zoo, you need to clone and unzip the datsets. Then, set the paths in the config to point to them.

```{bash}
/zoo/__config__.py
```

Different models from the zoo may need additional packages. For mine, you will also need

```{bash}
conda install torchaudio cpuonly -c pytorch -y
pip install pystoi glob2

```

Get the conda kernel to show in your jupyter session with

```{bash}
conda install -yn jaxenv ipykernel 
ipython kernel install --user --name=jaxenv
```

Finally, if you want to run nara or speex, and use FWSSNR do:

```{bash}
pip install nara_wpe
```

then to get FWSSNR

```{bash}
pip3 install https://github.com/schmiph2/pysepm/archive/master.zip
```

and get speex from <https://github.com/xiongyihui/speexdsp-python> and install it with

```{bash}
sudo apt install libspeexdsp-dev
git clone https://github.com/xiongyihui/speexdsp-python.git
cd speexdsp-python
python setup.py install
pip install speexdsp
```

## Data Download

### Acoustic Echo Cancellation

Follow the instructions [here](https://github.com/microsoft/AEC-Challenge) to get and download the Microsoft acoustic echo cancellation challenge dataset. Unzip the dataset and set the base path for `AEC_DATA_DIR` in `/zoo/__config__.py`. Then, for the RIRs, download the dataset from [here](https://www.openslr.org/28/) and set the `RIR_DATA_DIR` in `/zoo/__config__.py`.

### Equalization

Download the dataset from [here](https://zenodo.org/record/4660670#.YlmuBpPMKYQ). Unzip and set the base path for `EQ_DATA_DIR` in `/zoo/__config__.py`.

### Dereverberation

Follow the instructions [here](http://reverb2014.dereverberation.com/download.html) to get and download the REVERB challenge dataset. Unzip the dataset and set the base path for `REVERB_DATA_DIR` in `/zoo/__config__.py`.

### GSC Beamforming

Follow the instructions [here](https://catalog.ldc.upenn.edu/LDC2017S24) to get and download the CHIME3 challenge dataset. We modified the dataset to also write out the echoic clean multi-channel speech. The dataset structure is therefore unmodifed except for the addition of `<FILE NAME>_speech.<CHANNEL #>.wav` for each file. Set the base path for `CHIME3_DATA_DIR` in `/zoo/__config__.py`.


## License

All core utility code within the `metaaf` folder is licensed via the [University of Illinois Open Source License](../metaaf/LICENSE). All code within the `zoo` folder and model weights are licensed via the [Adobe Research License](LICENSE). Copyright (c) Adobe Systems Incorporated. All rights reserved.