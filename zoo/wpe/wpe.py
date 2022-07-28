import argparse
import os
import glob2

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset

import metaaf
from metaaf.data import NumpyLoader
from metaaf.filter import OverlapAdd
from metaaf.meta import MetaAFTrainer
from metaaf.callbacks import CheckpointCallback, WandBCallback, AudioLoggerCallback
from metaaf import optimizer_fgru

from zoo import metrics
from zoo.__config__ import REVERB_DATA_DIR


def numpy_trim_collate(batch):
    if isinstance(batch[0], np.ndarray):
        # max_len = max(map(lambda x: x.shape[0], batch))
        # batch = map(lambda x: wrap_pad(x, max_len), batch)
        min_len = min(map(lambda x: x.shape[0], batch))
        batch = map(lambda x: x[:min_len], batch)
        return np.stack(list(batch))
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_trim_collate(samples) for samples in transposed]
    elif isinstance(batch[0], dict):
        return dict(
            (key, numpy_trim_collate([d[key] for d in batch])) for key in batch[0]
        )
    else:
        return np.array(batch)


class ReverbDataset(Dataset):
    def __init__(
        self,
        base_dir=REVERB_DATA_DIR,
        mode="train",
        n_mics=8,
        signal_len=None,
        debug=False,
    ):
        self.mode = mode
        self.max_n_mics = 8
        self.n_mics = n_mics
        self.signal_len = signal_len
        self.debug = debug

        if self.mode == "train":
            self.base_dir = os.path.join(base_dir, "REVERB_WSJCAM0_tr", "data")
            clean_dir = os.path.join(self.base_dir, "mc_train_clean")

            # collect the dry_files first
            self.dry_files = glob2.glob(clean_dir + "/**/*.wav")

            # get the wet for each dry
            self.wet_files = [self.get_wet_tr(f) for f in self.dry_files]

        # double lenth of the dry test files for the primary and secodary mics
        elif self.mode == "val":
            self.base_dir = os.path.join(base_dir, "REVERB_WSJCAM0_dt", "data")
            dry_dir = os.path.join(self.base_dir, "cln_test")

            # collect the dry files
            dry_files = glob2.glob(dry_dir + "/**/*.wav")

            # match them once with near and once with far
            is_near = [False] * len(dry_files) + [True] * len(dry_files)
            self.dry_files = dry_files + dry_files

            # get the wet ones
            self.wet_files = [
                self.get_wet_et_dt(f, is_near[i]) for i, f in enumerate(self.dry_files)
            ]

        elif self.mode == "test":
            self.base_dir = os.path.join(base_dir, "REVERB_WSJCAM0_et", "data")
            dry_dir = os.path.join(self.base_dir, "cln_test")
            # collect the dry files
            dry_files = glob2.glob(dry_dir + "/**/*.wav")

            # match them once with near and once with far
            is_near = [False] * len(dry_files) + [True] * len(dry_files)
            self.dry_files = dry_files + dry_files

            # get the wet ones
            self.wet_files = [
                self.get_wet_et_dt(f, is_near[i]) for i, f in enumerate(self.dry_files)
            ]

    def __len__(self):
        if self.debug:
            return 512
        return len(self.dry_files)

    def get_wet_tr(self, dry_f_name):
        base, fname = os.path.split(dry_f_name)
        fname = os.path.splitext(fname)[0]
        base, speaker_dir = os.path.split(base)
        base, mic_dir_1 = os.path.split(base)
        base, mic_dir_2 = os.path.split(base)

        wet_dir = os.path.join(
            self.base_dir, "mc_train", mic_dir_2, mic_dir_1, speaker_dir
        )

        wets = [
            os.path.join(wet_dir, f"{fname}_ch{i + 1}.wav")
            for i in range(self.max_n_mics)
        ]
        return wets

    def get_wet_et_dt(self, dry_f_name, is_near):
        base, fname = os.path.split(dry_f_name)
        fname = os.path.splitext(fname)[0]
        base, speaker_dir = os.path.split(base)
        base, mic_dir_1 = os.path.split(base)
        base, mic_dir_2 = os.path.split(base)

        if is_near:
            wet_dir = os.path.join(
                self.base_dir, "near_test", mic_dir_2, mic_dir_1, speaker_dir
            )
        else:
            wet_dir = os.path.join(
                self.base_dir, "far_test", mic_dir_2, mic_dir_1, speaker_dir
            )

        wets = [
            os.path.join(wet_dir, f"{fname}_ch{i + 1}.wav")
            for i in range(self.max_n_mics)
        ]
        return wets

    def load_wav(self, f_names):
        if isinstance(f_names, list):
            x = np.array([sf.read(f)[0] for f in f_names]).T
        else:
            x = np.array(sf.read(f_names)[0])[:, None]
        return x

    def trim_wrap_pad_wav(self, wavs, trim_is_random=False):
        if self.signal_len is not None:
            if wavs.shape[0] > self.signal_len:
                if trim_is_random:
                    start = np.random.randint(0, len(wavs) - self.signal_len)
                    wavs = wavs[start : start + self.signal_len]
                else:
                    wavs = wavs[: self.signal_len]
            else:
                wavs = np.pad(
                    wavs, ((0, self.signal_len - wavs.shape[0]), (0, 0)), "wrap"
                )

        return wavs

    def __getitem__(self, idx):
        if self.debug:
            idx = idx % 8

        dry_f_name = self.dry_files[idx]
        wet_f_names = self.wet_files[idx]

        # randomlyu grab channels in training but just take first at val/test
        if self.mode == "train":
            chan_idxs = np.random.choice(self.max_n_mics, self.n_mics, replace=False)
            wet_f_names = [wet_f_names[chan_idx] for chan_idx in chan_idxs]
        else:
            wet_f_names = wet_f_names[: self.n_mics]

        # load the files
        dry_wav = self.load_wav(dry_f_name)
        wet_wav = self.load_wav(wet_f_names)

        # randomly trim at training/val but leave untouched(throws error o/w) at test
        if self.mode == "train":
            dry_wav = self.trim_wrap_pad_wav(dry_wav, trim_is_random=True)
            wet_wav = self.trim_wrap_pad_wav(wet_wav, trim_is_random=True)
        elif self.mode == "val":
            dry_wav = self.trim_wrap_pad_wav(dry_wav, trim_is_random=False)
            wet_wav = self.trim_wrap_pad_wav(wet_wav, trim_is_random=False)

        return {"signals": {"u": dry_wav, "d": wet_wav}, "metadata": {}}


class WPEOLA(OverlapAdd, hk.Module):
    def __init__(self, n_taps, delay, **kwargs):
        super().__init__(**kwargs)
        self.n_taps = n_taps
        self.delay = delay

        assert self.n_frames == self.n_taps + self.delay

        # select the analysis and synthesis windows
        self.analysis_window = jnp.hanning(self.window_size + 1)[:-1] ** 0.5
        self.synthesis_window = self.get_synthesis_window(self.analysis_window)

    def stable_normalized_loss(self, out, d):
        power = (d.conj() * d).real.mean((0, 2))
        eps = 1e-3 * jnp.max(power)

        inverse_power = 1 / jnp.maximum(power, eps)

        return ((out.conj() * out).real * inverse_power).mean()

    def __ola_call__(self, d, metadata):
        # collect a buffer sized anti-aliased filter
        w = self.get_filter(name="w", shape=(self.n_taps, self.n_freq, self.n_in_chan))

        # make filter inputs and targets from buffer
        d_target = d[-1, :, 0]
        d_input = d[: self.n_taps]

        # apply the filter
        d_est = (w.conj() * d_input).sum((0, 2))
        out = d_target - d_est
        return {
            "out": out[..., None],
            "u": d_input,
            "d": d_target[None, ..., None],
            "e": (d_target - d_est)[None, ..., None],
            "loss": self.stable_normalized_loss(out, d),
        }

    @staticmethod
    def add_args(parent_parser):
        parent_parser = super(WPEOLA, WPEOLA).add_args(parent_parser)
        parser = parent_parser.add_argument_group("DereverbOLA")
        parser.add_argument("--n_taps", type=int, default=10)
        parser.add_argument("--delay", type=int, default=3)
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = ["n_taps", "delay"]
        class_keys = {k: kwargs[k] for k in keys}
        class_keys.update(super(WPEOLA, WPEOLA).grab_args(kwargs))
        class_keys["n_frames"] = class_keys["n_taps"] + class_keys["delay"]
        return class_keys


def _WPEOLA_fwd(d, u, metadata=None, init_data=None, **kwargs):
    gen_filter = WPEOLA(**kwargs)
    return gen_filter(d=d)


def dereverb_loss(out, data_samples, metadata):
    return out["loss"]


def simple_stft(x, window_size=512, hop_size=256):
    x = jnp.pad(x, ((0, window_size), (0, 0)))
    n_frames = (len(x) - window_size) // hop_size
    window_idx = jnp.arange(window_size)[None, :]
    frame_idx = jnp.arange(n_frames)[:, None]
    window_idxs = window_idx + frame_idx * hop_size

    # index the buffer with the map and window
    windowed_x = x[window_idxs] * jnp.hanning(window_size)[None, :, None]

    # 0 is T, 1 will be F
    stft_x = jnp.fft.rfft(windowed_x, axis=1) / jnp.sqrt(window_size)
    return stft_x


def make_srr_val(window_size, hop_size):
    buffer_size = window_size - hop_size

    def srr_val(losses, outputs, data_samples, metadata, outer_learnable):
        srr_scores = []
        for i in range(len(outputs)):
            out_trim = outputs[i, buffer_size:]
            min_len = min(out_trim.shape[0], data_samples["d"].shape[1])

            d = data_samples["d"][i, :min_len, 0]
            y = out_trim[:min_len, 0]

            srr = metrics.srr_stft(np.array(y), np.array(d))
            srr_scores.append(srr)
        return jnp.mean(jnp.array(srr_scores))

    return srr_val


def make_meta_log_mse_loss(window_size, hop_size):
    buffer_size = window_size - hop_size
    EPS = 1e-8

    def meta_log_mse_loss(losses, outputs, data_samples, metadata, outer_learnable):
        # trim the buffer
        out_trim = outputs[buffer_size:]
        d_trim = data_samples["d"][: len(out_trim)]

        stft_output = jnp.mean(jnp.abs(simple_stft(out_trim)) ** 2, (0, 2))
        stft_d = jnp.mean(jnp.abs(simple_stft(d_trim)) ** 2, (0, 2))
        eps = 1e-3 * jnp.max(stft_d)
        inverse_power = 1 / jnp.maximum(stft_d, eps)

        return jnp.log(jnp.mean(stft_output * inverse_power) + EPS)

    return meta_log_mse_loss


def make_stoi_val(window_size, hop_size):
    buffer_size = window_size - hop_size

    def stoi_val(losses, outputs, data_samples, outer_learnable):
        stoi_scores = []
        for i in range(len(outputs)):
            out_trim = outputs[i, buffer_size:]
            min_len = min(out_trim.shape[0], data_samples["u"].shape[1])

            u = data_samples["u"][i, :min_len, 0]
            y = out_trim[:min_len, 0]

            stoi = metrics.stoi(np.array(y), np.array(u))
            stoi_scores.append(stoi)
        return -jnp.mean(jnp.array(stoi_scores))

    return stoi_val


"""
WPE 5 1
python wpe.py --n_frames 7 --window_size 512 --hop_size 256 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_wpe_5_1_c --unroll 16 --n_taps 5 --delay 2 --optimizer fgru

WPE 5 4
python wpe.py --n_frames 7 --window_size 512 --hop_size 256 --n_in_chan 4 --n_out_chan 1 --is_real --n_devices 1 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_wpe_5_4_c --unroll 16 --n_taps 5 --delay 2 --optimizer fgru 

WPE 5 8
python wpe.py --n_frames 7 --window_size 512 --hop_size 256 --n_in_chan 8 --n_out_chan 1 --is_real --n_devices 1 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_wpe_5_8_c_td --unroll 16 --n_taps 5 --delay 2 --optimizer fgru 
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser = optimizer_fgru.TimeChanCoupledGRU.add_args(parser)

    parser = WPEOLA.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())

    # set outsize automatically
    kwargs["outsize"] = kwargs["n_in_chan"] * kwargs["n_taps"]

    outer_train_loss = make_meta_log_mse_loss(kwargs["window_size"], kwargs["hop_size"])
    outer_val_loss = make_srr_val(kwargs["window_size"], kwargs["hop_size"])
    pprint.pprint(kwargs)

    # make the dataloders
    train_loader = NumpyLoader(
        ReverbDataset(
            mode="train",
            n_mics=kwargs["n_in_chan"],
            signal_len=128000,
            debug=kwargs["debug"],
        ),
        batch_size=kwargs["batch_size"],
        collate_fn=numpy_trim_collate,
        drop_last=True,
        shuffle=True,
        num_workers=16,
    )

    val_loader = NumpyLoader(
        ReverbDataset(
            mode="val",
            n_mics=kwargs["n_in_chan"],
            signal_len=128000,
            debug=kwargs["debug"],
        ),
        batch_size=kwargs["batch_size"] // kwargs["n_devices"],
        collate_fn=numpy_trim_collate,
        drop_last=True,
        num_workers=8,
    )

    # THIS NEEDS TO BE BATCH SIZE ONE AND NOT TRIM/PAD THE TEST DATA
    test_loader = NumpyLoader(
        ReverbDataset(
            mode="test",
            n_mics=kwargs["n_in_chan"],
            debug=kwargs["debug"],
        ),
        batch_size=1,
        num_workers=0,
    )
    # make the callbacks
    callbacks = [
        CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./taslp_ckpts"),
        AudioLoggerCallback(name=kwargs["name"], outputs_base_dir="./taslp_outputs"),
        WandBCallback(project="meta-wpe", name=kwargs["name"], entity=None),
    ]

    system = MetaAFTrainer(
        _filter_fwd=_WPEOLA_fwd,
        filter_kwargs=WPEOLA.grab_args(kwargs),
        filter_loss=dereverb_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=optimizer_fgru._timechancoupled_gru_fwd,
        optimizer_kwargs=optimizer_fgru.TimeChanCoupledGRU.grab_args(kwargs),
        meta_train_loss=outer_train_loss,
        meta_val_loss=outer_val_loss,
        init_optimizer=optimizer_fgru.init_optimizer_all_data,
        make_mapped_optmizer=optimizer_fgru.make_mapped_optmizer_all_data,
        callbacks=callbacks,
        kwargs=kwargs,
    )

    # start the training
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(
        **MetaAFTrainer.grab_args(kwargs),
        count_first_val=False,
        meta_opt_kwargs={"step_size": 1e-4, "b1": 0.99},
        key=key,
    )
