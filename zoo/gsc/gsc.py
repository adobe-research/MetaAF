import numpy as np
import argparse
import soundfile as sf
from pathlib import Path
import os

import jax
from jax import jit
import jax.numpy as jnp
from jax.tree_util import Partial
import haiku as hk
from torch.utils.data import Dataset

import metaaf
from metaaf.data import NumpyLoader
from metaaf.filter import OverlapAdd
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_fgru
from metaaf.callbacks import CheckpointCallback, WandBCallback, AudioLoggerCallback

from zoo import metrics
from zoo.__config__ import CHIME3_DATA_DIR


class Chime3Dataset(Dataset):
    def __init__(
        self,
        base_dir=CHIME3_DATA_DIR,
        mode="train",
        n_mics=6,
        signal_len=None,
        dynamic_noise_interfere=False,
        dynamic_speech_interfere=False,
        static_speech_interfere=False,
        debug=False,
        max_n_files=None,
    ):
        self.mode = mode
        self.n_mics = n_mics
        self.signal_len = signal_len
        self.static_speech_interfere = static_speech_interfere
        self.dynamic_noise_interfere = dynamic_noise_interfere
        self.dynamic_speech_interfere = dynamic_speech_interfere
        self.debug = debug
        self.max_n_files = max_n_files

        places = ["bus", "caf", "ped", "str"]
        mix_suffix = "simu"
        base_dir = os.path.join(base_dir, "data/audio/16kHz/isolated")

        if self.mode == "train":
            prefix = "tr05"
            clean_suffix = "org"
            self.clean_files, self.mix_files = self.get_files(
                base_dir, prefix, places, clean_suffix, mix_suffix, f_type=".wav"
            )

        elif self.mode == "val":
            prefix = "dt05"
            clean_suffix = "bth"
            self.clean_files, self.mix_files = self.get_files(
                base_dir, prefix, places, clean_suffix, mix_suffix, f_type=".CH1.wav"
            )

        elif self.mode == "test":
            prefix = "et05"
            clean_suffix = "bth"
            self.clean_files, self.mix_files = self.get_files(
                base_dir, prefix, places, clean_suffix, mix_suffix, f_type=".CH1.wav"
            )

        if (
            self.dynamic_noise_interfere
            or self.dynamic_speech_interfere
            or self.static_speech_interfere
        ):
            if mode == "val":
                np.random.seed(95)
                pair1 = np.arange(len(self.clean_files))
                np.random.shuffle(pair1)
                pair2 = np.arange(len(self.clean_files))
                np.random.shuffle(pair2)

                self.interfere_change_pair = np.array([pair1, pair2]).T
                self.interfere_change_spots = np.random.uniform(
                    0.1, 0.9, size=len(self.clean_files)
                )

            elif mode == "test":
                np.random.seed(1337)
                pair1 = np.arange(len(self.clean_files))
                np.random.shuffle(pair1)
                pair2 = np.arange(len(self.clean_files))
                np.random.shuffle(pair2)

                self.interfere_change_pair = np.array([pair1, pair2]).T
                self.interfere_change_spots = np.random.uniform(
                    0.1, 0.9, size=len(self.clean_files)
                )

    def __len__(self):
        if self.debug:
            return 512
        if self.max_n_files is not None:
            return self.max_n_files
        return len(self.mix_files)

    def get_files(
        self, base_dir, prefix, places, clean_suffix, mix_suffix, f_type=".wav"
    ):
        mix_ch1_files = []
        for place in places:
            p = Path(os.path.join(base_dir, f"{prefix}_{place}_{mix_suffix}"))
            all_files = list(p.glob("./*.CH1.wav"))
            speech_files = list(p.glob("./*_speech.CH1.wav"))
            mix_files = list(set(all_files) - set(speech_files))
            mix_ch1_files.extend(mix_files)

        mix_ch1_files.sort()

        clean_files = []
        mix_files = []
        for mix_file in mix_ch1_files:
            base_ch_name = str(mix_file)[:-8] + "_speech"
            clean_files.append(
                [os.path.join(base_ch_name + f".CH{i}.wav") for i in range(1, 7)]
            )

            base_ch_name = mix_file.stem[:-1]
            mix_files.append(
                [
                    os.path.join(mix_file.parent, base_ch_name + f"{i}.wav")
                    for i in range(1, 7)
                ]
            )
        return clean_files, mix_files

    def load_wav(self, f_names):
        if isinstance(f_names, list):
            x = np.array([sf.read(f)[0] for f in f_names]).T
        else:
            x = np.array(sf.read(f_names)[0])[:, None]
        return x

    def trim_wrap_pad_wav(self, wavs, signal_len):
        if wavs.shape[0] > signal_len:
            wavs = wavs[:signal_len]
        else:
            wavs = np.pad(wavs, ((0, signal_len - wavs.shape[0]), (0, 0)), "wrap")

        return wavs

    def load_from_idx(self, idx):
        if self.debug:
            idx = idx % 8

        mix_f_name = self.mix_files[idx]
        clean_f_name = self.clean_files[idx]

        for f in mix_f_name:
            assert os.path.exists(f)
        mix_wav = self.load_wav(mix_f_name)[:, : self.n_mics]
        clean_wav = self.load_wav(clean_f_name)[:, : self.n_mics]

        if self.signal_len is not None:
            mix_wav = self.trim_wrap_pad_wav(mix_wav, self.signal_len)
            clean_wav = self.trim_wrap_pad_wav(clean_wav, self.signal_len)

        return {"m": mix_wav, "s": clean_wav}

    def __getitem__(self, idx):
        data_dict = self.load_from_idx(idx)

        if self.dynamic_noise_interfere or self.dynamic_speech_interfere:
            # load the next mixture and trim/wrap to shape
            if self.mode == "train":
                interfere_change_pair_1 = np.random.randint(len(self.clean_files))
                interfere_change_pair_2 = np.random.randint(len(self.clean_files))
                interfere_change_spot = np.random.uniform(0.1, 0.9)
            else:
                interfere_change_pair_1 = self.interfere_change_pair[idx, 0]
                interfere_change_pair_2 = self.interfere_change_pair[idx, 1]
                interfere_change_spot = self.interfere_change_spots[idx]

            spot_1 = self.load_from_idx(interfere_change_pair_1)
            m_1 = self.trim_wrap_pad_wav(spot_1["m"], len(data_dict["m"]))
            s_1 = self.trim_wrap_pad_wav(spot_1["s"], len(data_dict["m"]))
            n_1 = m_1 - s_1

            spot_2 = self.load_from_idx(interfere_change_pair_2)
            m_2 = self.trim_wrap_pad_wav(spot_2["m"], len(data_dict["m"]))
            s_2 = self.trim_wrap_pad_wav(spot_2["s"], len(data_dict["m"]))
            n_2 = m_2 - s_2

            interfere = np.zeros_like(data_dict["m"])
            switch_spot = int(interfere_change_spot * len(interfere))
            if self.dynamic_noise_interfere:
                interfere[:switch_spot] += n_1[:switch_spot]
                interfere[switch_spot:] += n_2[switch_spot:]
            if self.dynamic_speech_interfere:
                interfere[:switch_spot] += s_1[:switch_spot]
                interfere[switch_spot:] += s_2[switch_spot:]

            # update the mixture with the new noise
            data_dict["m"] = data_dict["s"] + interfere

        if self.static_speech_interfere:
            if self.mode == "train":
                interfere_change_pair = np.random.randint(len(self.clean_files))
            else:
                interfere_change_pair = self.interfere_change_pair[idx, 0]
            next_data_dict = self.load_from_idx(interfere_change_pair)
            s = self.trim_wrap_pad_wav(next_data_dict["s"], len(data_dict["m"]))

            # update the mixture with the new noise
            data_dict["m"] = data_dict["s"] + s

        return {"signals": data_dict, "metadata": {}}


class GSCOLA(OverlapAdd, hk.Module):
    def __init__(
        self,
        exp_avg,
        cov_init,
        cov_update,
        cov_update_regularizer=0.0,
        steer_method="rank1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # select the analysis window
        self.analysis_window = jnp.hanning(self.window_size + 1)[:-1] ** 0.5
        self.synthesis_window = self.get_synthesis_window(self.analysis_window)
        self.exp_avg = exp_avg
        self.cov_init = cov_init
        self.cov_update = cov_update
        self.cov_update_regularizer = cov_update_regularizer

        if steer_method == "rank1":
            self.get_steering_vector = self.get_steering_r1
        elif steer_method == "eigh":
            self.get_steering_vector = self.get_steering_eigh

        if self.cov_init == "identity":
            self.cov_init_f = self.identity_cov_init
        elif self.cov_init in ["oracle", "mixture"]:
            self.cov_init_f = self.data_cov_init

    @staticmethod
    def data_cov_init(shape, dtype, S):
        phi = jnp.einsum("tfm,tfn->fmn", S, S.conj())
        return phi / S.shape[0]

    @staticmethod
    def identity_cov_init(shape, dtype):
        f_bins = shape[0]
        chans = shape[1]
        return jnp.stack(
            [jnp.identity(chans, dtype=jnp.complex64) for a in range(f_bins)]
        )

    @staticmethod
    @jit
    def update_spatial_cov(old_phi, m, s_hat, l, r):
        # calculate the covariance per freq
        phi = jnp.einsum("tfm,tfn->fmn", s_hat, s_hat.conj())
        phi = phi / s_hat.shape[0]

        # construct identity matrix for regularization
        identity = jnp.stack(
            [jnp.identity(phi.shape[1], dtype=phi.dtype) for a in range(phi.shape[0])]
        )

        # add regularizer
        new_phi = phi + r * identity

        # return update
        return l * old_phi + (1 - l) * new_phi

    @staticmethod
    @jit
    def get_steering_eigh(phi):
        # get the first eigenvalue normalized
        W, V = jnp.linalg.eigh(phi)
        idxs = jnp.argmax(jnp.abs(W), -1)
        v = V[jnp.arange(V.shape[0]), :, idxs]
        v = v * jnp.exp(-1j * jnp.angle(v[:, 0, None]))
        v = v / jnp.sqrt((v.conj() * v).real.sum(-1) + 1e-10)[:, None]
        return v

    @staticmethod
    @jit
    def get_steering_r1(phi):
        # get the first column normalized
        v = phi[:, :, 0]
        v = v * jnp.exp(-1j * jnp.angle(v[:, 0, None]))
        v = v / jnp.sqrt((v.conj() * v).real.sum(-1) + 1e-10)[:, None]
        return v

    @staticmethod
    @jit
    def get_blocking_matrix(phi):
        # empty blocking matrix
        B = jnp.zeros((phi.shape[0], phi.shape[1], phi.shape[1] - 1), dtype=phi.dtype)

        # channel 0 target transfer function
        X = phi[:, :, 0]
        X_0 = phi[:, 0, 0, None] + 1e-6

        B = B.at[:, 0].set(-X[:, 1:].conj() / X_0.conj())
        return B.at[:, 1:, :].set(jnp.eye(phi.shape[1] - 1))

    def __ola_call__(self, s, m):
        w = self.get_filter(name="w", shape=[1, self.n_freq, self.n_in_chan - 1])[0]

        # all GSC steps assume ref channel is at index 0
        phi = hk.get_state(
            "phi",
            (self.n_freq, self.n_in_chan, self.n_in_chan),
            init=self.cov_init_f,
        )

        s_hat = s if self.cov_update == "oracle" else m
        phi = self.update_spatial_cov(
            phi, m, s_hat, self.exp_avg, self.cov_update_regularizer
        )
        hk.set_state("phi", phi)

        v = self.get_steering_vector(phi)  # freq. x mics
        B = self.get_blocking_matrix(phi)  # freq. x mics x mics - 1

        # apply steering vector to current frame -> freq.
        x_steer = jnp.einsum("fm,fm->f", v.conj(), m[-1, :, :])

        # apply blocking matrix to current frame -> freq. x mics - 1
        x_block = jnp.einsum("fmn,fm->fn", B.conj(), m[-1, :, :])

        # apply adaptive filter on the output of the blocking matrix
        x_noise = jnp.einsum("fm,fm->f", w.conj(), x_block)

        out = (x_steer - x_noise)[:, None]  # freqs. x 1

        return {
            "out": out,
            "u": x_block[None],
            "d": x_steer[None, ..., None],
            "e": (x_steer - x_noise)[None, ..., None],
            "loss": jnp.vdot(out, out).real / out.size,
            "w": v - jnp.einsum("fmn,fn->fm", B, w),
        }

    def __call__(self, init_data, metadata=None, **kwargs):
        # collect buffered inputs for all inputs
        kwargs_buffer = {
            k: self.buffered_stft_analysis(v, self.analysis_window, k)
            for k, v in kwargs.items()
        }

        if init_data is not None and self.cov_init in ["oracle", "mixture"]:
            s = init_data["s"] if self.cov_init == "oracle" else init_data["m"]
            n_frames = (s.shape[0] - self.window_size) // self.hop_size
            S = self.stft_analysis(
                s,
                self.analysis_window,
                self.window_size,
                self.hop_size,
                self.pad_size,
                n_frames,
            )

            self.cov_init_f = Partial(self.cov_init_f, S=S)

        # call the users filtering function
        out = self.__ola_call__(**kwargs_buffer)

        # only ols first model output
        if isinstance(out, dict):
            # get non overlap added time-domain signal
            out["out"] = self.stft_synthesis(out["out"], self.synthesis_window)

            return out
        else:
            return self.stft_synthesis(out, self.synthesis_window)

    @staticmethod
    def add_args(parent_parser):
        parent_parser = super(GSCOLA, GSCOLA).add_args(parent_parser)
        parser = parent_parser.add_argument_group("DereverbOLA")
        parser.add_argument("--exp_avg", type=float, default=0.9)
        parser.add_argument("--cov_init", type=str, default="identity")
        parser.add_argument("--cov_update", type=str, default="oracle")
        parser.add_argument("--steer_method", type=str, default="rank1")
        parser.add_argument("--cov_update_regularizer", type=float, default=0.0)
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = [
            "exp_avg",
            "cov_init",
            "cov_update",
        ]
        class_keys = {k: kwargs[k] for k in keys}

        # post hoc add the steering
        if "steer_method" in kwargs:
            class_keys["steer_method"] = kwargs["steer_method"]
        else:
            class_keys["steer_method"] = "rank1"

        if "cov_update_regularizer" in kwargs:
            class_keys["cov_update_regularizer"] = kwargs["cov_update_regularizer"]
        else:
            class_keys["cov_update_regularizer"] = 0.0

        class_keys.update(super(GSCOLA, GSCOLA).grab_args(kwargs))
        return class_keys


def _GSCOLA_fwd(s, m, metadata=None, init_data=None, **kwargs):
    gen_filter = GSCOLA(**kwargs)
    return gen_filter(s=s, m=m, init_data=init_data)


def gsc_loss(out, data_samples, metadata):
    return out["loss"]


def meta_log_mse_loss(losses, outputs, data_samples, metadata, outer_learnable):
    EPS = 1e-8
    return jnp.log(jnp.mean(jnp.abs(outputs) ** 2) + EPS)


def make_neg_sisdr_val(window_size, hop_size):
    def neg_sisdr_val(losses, outputs, data_samples, metadata, outer_learnable):
        sisdr_scores = []
        buffer_size = window_size - hop_size

        for i in range(len(outputs)):
            out_trim = outputs[i, buffer_size:]
            min_len = min(out_trim.shape[0], data_samples["s"].shape[1])

            clean = data_samples["s"][i, :min_len, 0]
            enhanced = out_trim[:min_len, 0]

            sisdr_scores.append(metrics.sisdr(enhanced, clean))
        return -jnp.mean(jnp.array(sisdr_scores))

    return neg_sisdr_val


"""
Diffuse Interference
python gsc.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_gsc_diffuse --unroll 16 --cov_init identity --cov_update oracle --exp_avg .9 --steer_method eigh --cov_update_regularizer 0.01

Directional Interference
python gsc.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_gsc_direct --unroll 16 --cov_init identity --cov_update oracle --exp_avg .9 --steer_method eigh --cov_update_regularizer 0.01 --static_speech_interfere
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--static_speech_interfere", action="store_true")
    parser.add_argument("--outer_loss", type=str, default="self_mse")
    parser = optimizer_fgru.TimeChanCoupledGRU.add_args(parser)

    parser = GSCOLA.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())

    # set outsize automatically
    kwargs["outsize"] = kwargs["n_in_chan"] - 1
    pprint.pprint(kwargs)

    # make the dataloders
    train_loader = NumpyLoader(
        Chime3Dataset(
            mode="train",
            n_mics=kwargs["n_in_chan"],
            signal_len=128000,
            static_speech_interfere=kwargs["static_speech_interfere"],
            debug=kwargs["debug"],
        ),
        batch_size=kwargs["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=6,
    )

    val_loader = NumpyLoader(
        Chime3Dataset(
            mode="val",
            n_mics=kwargs["n_in_chan"],
            signal_len=128000,
            static_speech_interfere=kwargs["static_speech_interfere"],
        ),
        batch_size=kwargs["batch_size"],
        drop_last=True,
        num_workers=2,
    )

    test_loader = NumpyLoader(
        Chime3Dataset(
            mode="test",
            n_mics=kwargs["n_in_chan"],
            static_speech_interfere=kwargs["static_speech_interfere"],
        ),
        batch_size=1,
        num_workers=0,
    )

    # make the callbacks
    callbacks = [
        CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./taslp_ckpts"),
        AudioLoggerCallback(name=kwargs["name"], outputs_base_dir="./taslp_outputs"),
        WandBCallback(project="meta-gsc", name=kwargs["name"], entity=None),
    ]

    system = MetaAFTrainer(
        _filter_fwd=_GSCOLA_fwd,
        filter_kwargs=GSCOLA.grab_args(kwargs),
        filter_loss=gsc_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=optimizer_fgru._timechancoupled_gru_fwd,
        optimizer_kwargs=optimizer_fgru.TimeChanCoupledGRU.grab_args(kwargs),
        meta_train_loss=meta_log_mse_loss,
        meta_val_loss=make_neg_sisdr_val(kwargs["window_size"], kwargs["hop_size"]),
        init_optimizer=optimizer_fgru.init_optimizer_all_data,
        make_mapped_optmizer=optimizer_fgru.make_mapped_optmizer_all_data,
        callbacks=callbacks,
        kwargs=kwargs,
    )

    # start the training
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(
        **MetaAFTrainer.grab_args(kwargs),
        meta_opt_kwargs={"step_size": 1e-4, "b1": 0.99},
        key=key,
    )
