import numpy as np
import argparse
import soundfile as sf
import os
import random
import glob
import scipy

import jax
import jax.numpy as jnp
import haiku as hk
from torch.utils.data import Dataset
import torch
import torchaudio
from torchaudio import sox_effects

import metaaf
from metaaf.data import NumpyLoader
from metaaf.filter import OverlapSave
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_gru
from metaaf.optimizer_gru import ElementWiseGRU
from metaaf.callbacks import CheckpointCallback, WandBCallback, AudioLoggerCallback

from zoo import metrics
from zoo.__config__ import EQ_DATA_DIR


class EQDAPSDataset(Dataset):
    def __init__(
        self,
        base_dir=EQ_DATA_DIR,
        signal_len=80000,
        mode="train",
        dry_file_splits=[0.6, 0.2, 0.2],
        n_signals=1000,
        is_fir=False,
    ):

        self.mode = mode
        self.base_dir = base_dir
        self.n_signals = n_signals
        self.sr = 16000
        self.is_fir = is_fir

        # Making the dataset
        # grab and partition the dry files for no data overlap
        random.seed(7)
        dry_files = glob.glob(os.path.join(base_dir, "cleanraw") + "/*.wav")
        random.shuffle(dry_files)

        n_train = int(len(dry_files) * dry_file_splits[0])
        n_val = int(len(dry_files) * dry_file_splits[1])
        n_test = int(len(dry_files) * dry_file_splits[2])

        if mode == "train":
            files = dry_files[:n_train]
        elif mode == "val":
            files = dry_files[n_train : n_train + n_val]
        else:
            files = dry_files[-n_test:]

        # for reproducible random params
        if mode == "train":
            random.seed(42)
            np.random.seed(42)
        elif mode == "val":
            random.seed(95)
            np.random.seed(95)
        else:
            random.seed(1337)
            np.random.seed(1337)

        # save parameters to randomly
        # 1. pick a dry file
        # 2. pick a signal_len chunk of dry file
        # 3. make a random eq filter

        self.data = []
        for i in range(n_signals):
            # pick the file
            file_idx = np.random.randint(0, len(files))

            # pick the chunks
            f = sf.SoundFile(files[file_idx])
            n_samples = f.frames
            file_sr = f.samplerate

            lenght_in_file_sr = int(signal_len / self.sr * file_sr)
            start_idx = np.random.randint(0, n_samples - lenght_in_file_sr)
            stop_idx = start_idx + lenght_in_file_sr

            # make the random effect
            n_effects = np.random.randint(5, 15)
            c = np.random.randint(1, 8000, size=n_effects)
            q = np.random.uniform(0.1, 10, size=n_effects)
            g = np.random.uniform(-18, 18, size=n_effects)

            effects = [
                ["equalizer", str(c[i]), str(q[i]), str(g[i])] for i in range(n_effects)
            ]

            cur_data = {
                "file": files[file_idx],
                "start": start_idx,
                "stop": stop_idx,
                "effects": effects,
            }
            self.data.append(cur_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d, sr = sf.read(
            self.data[idx]["file"],
            start=self.data[idx]["start"],
            stop=self.data[idx]["stop"],
        )

        d_torch = torch.tensor(d.astype("float32"))
        d_torch = torchaudio.transforms.Resample(sr, self.sr)(d_torch)[:, None]

        if self.is_fir:
            # get the numpy impulse response
            delta_torch = torch.zeros(512, 1)
            delta_torch[0] = 1

            h = torchaudio.sox_effects.apply_effects_tensor(
                delta_torch, 16000, self.data[idx]["effects"], channels_first=False
            )[0].numpy()
            d = np.array(d_torch[:, 0])
            u = scipy.signal.fftconvolve(d, h[:, 0])[: len(d)]
            return {"signals": {"d": d[:, None], "u": u[:, None]}, "metadata": {}}

        else:
            u_torch, sr = sox_effects.apply_effects_tensor(
                d_torch, self.sr, self.data[idx]["effects"], channels_first=False
            )

            d = np.array(d_torch)
            u = np.array(u_torch)
            return {"signals": {"d": d, "u": u}, "metadata": {}}


class EQOLS(OverlapSave, hk.Module):
    def __init__(self, constraint, **kwargs):
        super().__init__(**kwargs)
        # select the analysis window
        self.analysis_window = jnp.ones(self.window_size)
        self.constraint = constraint

    def __ols_call__(self, u, d, metadata):
        w = hk.get_parameter(
            "w",
            [self.n_frames, self.n_freq, self.n_in_chan],
            init=self.w_init,
            dtype=jnp.complex64,
        )

        if self.constraint == "antialias":
            # time domain antialias
            w_td = (
                self.ifft(w, axis=1)
                .at[:, (self.window_size + self.pad_size) // 2 :, :]
                .set(0.0)
            )
            w = self.fft(w_td, axis=1)

        d_hat = (w * u).sum(0)
        e = d[-1] - d_hat

        return {
            "out": d_hat,
            "u": u[-1, None],
            "d": d,
            "e": e[None],
            "loss": jnp.vdot(e, e).real / e.size,
        }

    @staticmethod
    def add_args(parent_parser):
        parent_parser = super(EQOLS, EQOLS).add_args(parent_parser)
        parser = parent_parser.add_argument_group("EqualizationOLS")
        parser.add_argument("--constraint", type=str, default="antialias")
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = ["constraint"]
        class_keys = {k: kwargs[k] for k in keys}
        class_keys.update(super(EQOLS, EQOLS).grab_args(kwargs))
        return class_keys


def _EQOLS_fwd(u, d, metadata=None, init_data=None, **kwargs):
    gen_filter = EQOLS(**kwargs)
    return gen_filter(u=u, d=d)


def eq_loss(out, data_samples, metadata):
    return out["loss"]


def meta_log_mse_loss(losses, outputs, data_samples, metadata, outer_learnable):
    EPS = 1e-8
    return jnp.log(jnp.mean(jnp.abs(outputs - data_samples["d"]) ** 2) + EPS)


def neg_snr_val_loss(losses, outputs, data_samples, metadata, outer_learnable):
    snr_scores = []
    for i in range(len(outputs)):
        min_len = min(outputs.shape[1], data_samples["u"].shape[1])

        d = data_samples["d"][i, :min_len, 0]
        d_hat = outputs[i, :min_len, 0]

        snr = metrics.snr(np.array(d_hat), np.array(d))
        snr_scores.append(snr)
    return -jnp.mean(jnp.array(snr_scores))


"""
Aliased
python eq.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_eq_none_16_c --unroll 16 --constraint none

Anti-Aliased
python eq.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_eq_antialias_16_c --unroll 16 --constraint antialias
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")

    parser = EQOLS.add_args(parser)
    parser = ElementWiseGRU.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    # make the dataloders
    train_loader = NumpyLoader(
        EQDAPSDataset(mode="train", is_fir=False, n_signals=16384),
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=8,
    )
    val_loader = NumpyLoader(
        EQDAPSDataset(mode="val", is_fir=False, n_signals=2048),
        batch_size=kwargs["batch_size"],
        num_workers=4,
    )
    test_loader = NumpyLoader(
        EQDAPSDataset(mode="test", is_fir=True, n_signals=2048),
        batch_size=kwargs["batch_size"],
        num_workers=0,
    )

    # make the callbacks
    callbacks = [
        CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./taslp_ckpts"),
        AudioLoggerCallback(name=kwargs["name"], outputs_base_dir="./taslp_outputs"),
        WandBCallback(project="taslp-eq", name=kwargs["name"], entity="autodsp"),
    ]

    system = MetaAFTrainer(
        _filter_fwd=_EQOLS_fwd,
        filter_kwargs=EQOLS.grab_args(kwargs),
        filter_loss=eq_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer_kwargs=ElementWiseGRU.grab_args(kwargs),
        meta_train_loss=meta_log_mse_loss,
        meta_val_loss=neg_snr_val_loss,
        init_optimizer=optimizer_gru.init_optimizer_all_data,
        make_mapped_optmizer=optimizer_gru.make_mapped_optmizer_all_data,
        callbacks=callbacks,
        kwargs=kwargs,
    )

    # start the training
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(
        **MetaAFTrainer.grab_args(kwargs),
        meta_opt_kwargs={"step_size": 1e-4, "b1": 0.99},
        key=key
    )
