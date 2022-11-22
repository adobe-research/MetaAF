import numpy as np
import scipy
import argparse
import soundfile as sf
import os
import glob2
import pandas

import jax
import jax.numpy as jnp
import haiku as hk
from torch.utils.data import Dataset

import metaaf
from metaaf import optimizer_utils
from metaaf import optimizer_gru
from metaaf.data import NumpyLoader
from metaaf.filter import OverlapSave
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_fgru
from metaaf.callbacks import CheckpointCallback, WandBCallback, AudioLoggerCallback


from zoo import metrics
from zoo.__config__ import AEC_DATA_DIR, RIR_DATA_DIR


class ComboAECDataset(Dataset):
    def __init__(
        self,
        aec_dir=AEC_DATA_DIR,
        rir_dir=RIR_DATA_DIR,
        mode="train",
        random_roll=False,
        random_level=False,
        max_len=160000,
    ):

        self.mode = mode
        self.datasets = [
            MSFTAECDataset_RIR(
                aec_dir=aec_dir,
                rir_dir=rir_dir,
                mode=mode,
                double_talk=False,
                scene_change=False,
                random_roll=random_roll,
                random_level=random_level,
                max_len=max_len,
            ),
            MSFTAECDataset_RIR(
                aec_dir=aec_dir,
                rir_dir=rir_dir,
                mode=mode,
                double_talk=True,
                scene_change=False,
                random_roll=random_roll,
                random_level=random_level,
                max_len=max_len,
            ),
            MSFTAECDataset_RIR(
                aec_dir=aec_dir,
                rir_dir=rir_dir,
                mode=mode,
                double_talk=True,
                scene_change=True,
                random_roll=random_roll,
                random_level=random_level,
                max_len=max_len,
            ),
            MSFTAECDataset(
                base_dir=aec_dir,
                mode=mode,
                double_talk=True,
                scene_change=True,
                random_roll=random_roll,
                random_level=random_level,
                max_len=max_len,
            ),
        ]
        self.lens = np.array([len(dset) for dset in self.datasets])
        self.cum_lens = np.cumsum(self.lens)
        self.total_len = self.lens.sum()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        dset_idx = np.searchsorted(self.cum_lens - 1, idx)
        offset = 0 if dset_idx == 0 else self.cum_lens[dset_idx - 1]
        return self.datasets[dset_idx][idx - offset]


class MSFTAECDataset(Dataset):
    def __init__(
        self,
        base_dir=AEC_DATA_DIR,
        mode="train",
        double_talk=False,
        scene_change=False,
        random_roll=False,
        random_level=False,
        fix_train_roll=False,
        max_len=160000,
        denoising=False,
    ):
        if denoising:
            assert double_talk == True, print(
                "Nearend talk must be present for denoising"
            )

        synthetic_dir = os.path.join(base_dir, "datasets/synthetic/")

        csv = pandas.read_csv(os.path.join(synthetic_dir, "meta.csv"))
        nearend_scale_dict = {}
        for _, row in csv.iterrows():
            fileid = row["fileid"]
            nearend_scale_dict[fileid] = row["nearend_scale"]
        self.nearend_scale_dict = nearend_scale_dict

        self.echo_signal_dir = os.path.join(synthetic_dir, "echo_signal")
        self.farend_speech_dir = os.path.join(synthetic_dir, "farend_speech")
        self.nearend_mic_dir = os.path.join(synthetic_dir, "nearend_mic_signal")
        self.nearend_speech_dir = os.path.join(synthetic_dir, "nearend_speech")
        self.max_len = max_len

        self.double_talk = double_talk
        self.scene_change = scene_change
        self.random_roll = random_roll
        self.random_level = random_level
        self.min_uw_scale = -10.0
        self.max_uw_scale = 10.0
        self.fix_train_roll = fix_train_roll
        self.denoising = denoising

        if self.scene_change:
            if mode == "val":
                np.random.seed(95)
                self.scene_change_pair = np.arange(500)
                np.random.shuffle(self.scene_change_pair)
                self.scene_change_idx = np.random.randint(64000, 96000, size=500)
            elif mode == "test":
                np.random.seed(1337)
                self.scene_change_pair = np.arange(500)
                np.random.shuffle(self.scene_change_pair)
                self.scene_change_idx = np.random.randint(64000, 96000, size=500)

        if self.random_roll:
            if mode == "train" and self.fix_train_roll:
                np.random.seed(42)
                self.random_rolls = np.random.randint(0, 160000, size=9000)
            elif mode == "val":
                np.random.seed(95)
                self.random_rolls = np.random.randint(0, 160000, size=500)
            else:
                np.random.seed(1337)
                self.random_rolls = np.random.randint(0, 160000, size=500)

        self.mode = mode
        if self.mode == "test":
            self.offset = 0
        elif self.mode == "val":
            self.offset = 500
        elif self.mode == "train":
            self.offset = 1000

    def __len__(self):
        if self.mode == "test":
            return 500
        elif self.mode == "val":
            return 500
        elif self.mode == "train":
            return 9000

    def load_from_idx(self, idx):
        idx = idx + self.offset
        if self.double_talk:
            d, sr = sf.read(
                os.path.join(self.nearend_mic_dir, f"nearend_mic_fileid_{idx}.wav")
            )
        else:
            d, sr = sf.read(
                os.path.join(self.echo_signal_dir, f"echo_fileid_{idx}.wav")
            )

        u, sr = sf.read(
            os.path.join(self.farend_speech_dir, f"farend_speech_fileid_{idx}.wav")
        )

        if self.random_level and self.mode == "train":
            u_scale = np.sqrt(
                10 ** (np.random.uniform(self.min_uw_scale, self.max_uw_scale) / 10)
            )
            u = u * u_scale

        e, sr = sf.read(os.path.join(self.echo_signal_dir, f"echo_fileid_{idx}.wav"))
        s, sr = sf.read(
            os.path.join(self.nearend_speech_dir, f"nearend_speech_fileid_{idx}.wav")
        )

        u = np.pad(u, (0, max(0, self.max_len - len(u))))
        d = np.pad(d, (0, max(0, self.max_len - len(d))))
        e = np.pad(e, (0, max(0, self.max_len - len(e))))
        s = np.pad(s, (0, max(0, self.max_len - len(s))))

        s = s * self.nearend_scale_dict[idx]

        if self.denoising:
            d = d - e

        if self.random_roll:
            if self.mode == "train":
                if self.fix_train_roll:
                    shift = self.random_rolls[idx - self.offset]
                else:
                    shift = np.random.randint(0, self.max_len)

            else:
                shift = self.random_rolls[idx - self.offset]

            u = np.roll(u, shift)
            d = np.roll(d, shift)
            e = np.roll(e, shift)
            s = np.roll(s, shift)

        return {"d": d[:, None], "u": u[:, None], "e": e[:, None], "s": s[:, None]}

    def __getitem__(self, idx):
        data_dict = self.load_from_idx(idx)

        if self.scene_change:
            scene_change_pair = (
                np.random.randint(0, 9000)
                if self.mode == "train"
                else self.scene_change_pair[idx]
            )
            next_data_dict = self.load_from_idx(scene_change_pair)
            for k, v in data_dict.items():
                change_idx = (
                    np.random.randint(64000, 96000)
                    if self.mode == "train"
                    else self.scene_change_idx[idx]
                )
                data_dict[k] = np.concatenate(
                    (v[:change_idx], next_data_dict[k][change_idx:]), axis=0
                )

        return {"signals": data_dict, "metadata": {}}


class MSFTAECDataset_RIR(Dataset):
    def __init__(
        self,
        aec_dir=AEC_DATA_DIR,
        rir_dir=RIR_DATA_DIR,
        mode="train",
        double_talk=False,
        scene_change=False,
        random_roll=False,
        random_level=False,
        rir_len=None,
        max_len=160000,
    ):

        aec_synthetic_dir = os.path.join(aec_dir, "datasets/synthetic/")
        rir_dir = os.path.join(rir_dir, "simulated_rirs/")

        self.farend_speech_dir = os.path.join(aec_synthetic_dir, "farend_speech")
        self.nearend_speech_dir = os.path.join(aec_synthetic_dir, "nearend_speech")

        self.double_talk = double_talk
        self.scene_change = scene_change
        self.random_roll = random_roll
        self.random_level = random_level
        self.max_len = max_len
        self.rir_len = rir_len

        self.min_uw_scale = -10.0
        self.max_uw_scale = 10.0

        # get same rir split every time
        self.rirs = glob2.glob(rir_dir + "/**/*.wav")
        rng = np.random.RandomState(0)
        rng.shuffle(self.rirs)

        if mode == "train":
            self.rirs = self.rirs[1000:]
        elif mode == "val":
            self.rirs = self.rirs[500:1000]
        elif mode == "test":
            self.rirs = self.rirs[:500]

        if self.double_talk:
            self.min_ser = -10
            self.max_ser = 10
            if mode == "val":
                np.random.seed(95)
                self.sers = np.random.uniform(self.min_ser, self.max_ser, size=500)
            elif mode == "test":
                np.random.seed(1337)
                self.sers = np.random.uniform(self.min_ser, self.max_ser, size=500)

        if self.scene_change:
            start = int(0.4 * self.max_len)
            stop = int(0.6 * self.max_len)
            if mode == "val":
                np.random.seed(95)
                self.scene_change_pair = np.arange(500)
                np.random.shuffle(self.scene_change_pair)
                self.scene_change_idx = np.random.randint(start, stop, size=500)
            elif mode == "test":
                np.random.seed(1337)
                self.scene_change_pair = np.arange(500)
                np.random.shuffle(self.scene_change_pair)
                self.scene_change_idx = np.random.randint(start, stop, size=500)

        if self.random_roll:
            if mode == "val":
                np.random.seed(95)
                self.random_rolls = np.random.randint(0, self.max_len, size=500)
            elif mode == "test":
                np.random.seed(1337)
                self.random_rolls = np.random.randint(0, self.max_len, size=500)

        self.mode = mode
        if self.mode == "test":
            self.offset = 0
        elif self.mode == "val":
            self.offset = 500
        elif self.mode == "train":
            self.offset = 1000

    def __len__(self):
        if self.mode == "test":
            return 500
        elif self.mode == "val":
            return 500
        elif self.mode == "train":
            return 9000

    def load_from_idx(self, idx):
        speech_idx = idx + self.offset
        rir_idx = np.random.randint(len(self.rirs)) if self.mode == "train" else idx
        w, _ = sf.read(self.rirs[rir_idx])

        if self.rir_len is not None:
            w = w[: self.rir_len]

        u, sr = sf.read(
            os.path.join(
                self.farend_speech_dir, f"farend_speech_fileid_{speech_idx}.wav"
            )
        )

        if self.random_level and self.mode == "train":
            w_scale = np.sqrt(
                10 ** (np.random.uniform(self.min_uw_scale, self.max_uw_scale) / 10)
            )
            w = w * w_scale
            u_scale = np.sqrt(
                10 ** (np.random.uniform(self.min_uw_scale, self.max_uw_scale) / 10)
            )
            u = u * u_scale

        e = scipy.signal.fftconvolve(u, w)[: len(u)]

        if self.double_talk:
            s = sf.read(
                os.path.join(
                    self.nearend_speech_dir, f"nearend_speech_fileid_{speech_idx}.wav"
                )
            )[0]
            ser = (
                np.random.uniform(self.min_ser, self.max_ser)
                if self.mode == "train"
                else self.sers[idx]
            )
            s_ser_scale = np.sqrt(
                np.abs(e**2).mean() * (10 ** (ser / 10)) / np.abs(s**2).mean()
            )
            s = s * s_ser_scale
        else:
            s = np.zeros_like(e)

        d = e + s

        u = np.pad(u, (0, max(0, self.max_len - len(u))), "wrap")
        d = np.pad(d, (0, max(0, self.max_len - len(d))), "wrap")
        e = np.pad(e, (0, max(0, self.max_len - len(e))), "wrap")
        s = np.pad(s, (0, max(0, self.max_len - len(s))), "wrap")

        if self.random_roll:
            shift = (
                np.random.randint(0, self.max_len)
                if self.mode == "train"
                else self.random_rolls[idx - self.offset]
            )
            u = np.roll(u, shift)
            d = np.roll(d, shift)
            e = np.roll(e, shift)
            s = np.roll(s, shift)

        return {"d": d[:, None], "u": u[:, None], "e": e[:, None], "s": s[:, None]}

    def __getitem__(self, idx):
        data_dict = self.load_from_idx(idx)

        if self.scene_change:
            scene_change_pair = (
                np.random.randint(0, 9000)
                if self.mode == "train"
                else self.scene_change_pair[idx]
            )
            next_data_dict = self.load_from_idx(scene_change_pair)
            for k, v in data_dict.items():
                change_idx = (
                    np.random.randint(64000, 96000)
                    if self.mode == "train"
                    else self.scene_change_idx[idx]
                )
                data_dict[k] = np.concatenate(
                    (v[:change_idx], next_data_dict[k][change_idx:]), axis=0
                )
        return {"signals": data_dict, "metadata": {}}


class AECOLS(OverlapSave, hk.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # select the analysis window
        self.analysis_window = jnp.ones(self.window_size)

    def antialias(self, x):
        x_td = self.ifft(x, axis=1).at[:, : -self.hop_size, :].set(0.0)
        return self.fft(x_td, axis=1)

    def __ols_call__(self, u, d, metadata):
        w = self.get_filter(name="w")
        y = (w * u).sum(0)

        out = d[-1] - y

        y = self.antialias(y[None])
        e = self.antialias(out[None])

        return {
            "out": out,
            "u": u,
            "d": d[-1, None],
            "e": e,
            "y": y,
            "grad": jnp.conj(u) * e,
            "loss": jnp.vdot(e, e).real / 2,
        }

    @staticmethod
    def add_args(parent_parser):
        return super(AECOLS, AECOLS).add_args(parent_parser)

    @staticmethod
    def grab_args(kwargs):
        return super(AECOLS, AECOLS).grab_args(kwargs)


def _AECOLS_fwd(u, d, e, s, metadata=None, init_data=None, **kwargs):
    gen_filter = AECOLS(**kwargs)
    return gen_filter(u=u, d=d)


class NOOPAECOLS(OverlapSave, hk.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __ols_call__(self, u, d, metadata):
        pass

    def __call__(self, u, d):
        shape = [self.n_frames, self.n_freq, self.n_in_chan]
        w = hk.get_parameter("w", shape, init=metaaf.complex_utils.complex_zeros)
        return {
            "out": u[-1, None],
            "u": w,
            "d": w[-1, None],
            "e": w[-1, None],
            "y": w[-1, None],
            "grad": w,
            "loss": 0.0,
        }

    @staticmethod
    def add_args(parent_parser):
        return super(NOOPAECOLS, NOOPAECOLS).add_args(parent_parser)

    @staticmethod
    def grab_args(kwargs):
        return super(NOOPAECOLS, NOOPAECOLS).grab_args(kwargs)


def _NOOPAECOLS_fwd(u, d, e, s, metadata=None, init_data=None, **kwargs):
    gen_filter = NOOPAECOLS(**kwargs)
    return gen_filter(u=u, d=d)


def aec_loss(out, data_samples, metadata):
    return out["loss"]


def meta_mse_loss(
    losses, outputs, data_samples, metadata, outer_index, outer_learnable
):
    out = jnp.concatenate(outputs["out"], 0)
    return jnp.mean(jnp.abs(out) ** 2)


def meta_log_mse_loss(
    losses, outputs, data_samples, metadata, outer_index, outer_learnable
):
    out = jnp.concatenate(outputs["out"], 0)
    EPS = 1e-8
    return jnp.log(jnp.mean(jnp.abs(out) ** 2) + EPS)


def neg_erle_val_loss(losses, outputs, data_samples, metadata, outer_learnable):
    out = jnp.reshape(
        outputs["out"],
        (outputs["out"].shape[0], -1, outputs["out"].shape[-1]),
    )
    erle_scores = []
    for i in range(len(out)):
        min_len = min(
            out.shape[1], data_samples["d"].shape[1], data_samples["e"].shape[1]
        )

        e = data_samples["e"][i, :min_len, 0]
        d = data_samples["d"][i, :min_len, 0]
        y = out[i, :min_len, 0]

        erle = metrics.erle(np.array(y), np.array(d), np.array(e))
        erle_scores.append(erle)
    return -jnp.mean(jnp.array(erle_scores))


def neg_serle_val_loss(losses, outputs, data_samples, metadata, outer_learnable):
    out = jnp.reshape(
        outputs["out"],
        (outputs["out"].shape[0], -1, outputs["out"].shape[-1]),
    )
    serle_scores = []
    for i in range(len(out)):
        min_len = min(
            out.shape[1], data_samples["d"].shape[1], data_samples["e"].shape[1]
        )

        e = data_samples["e"][i, :min_len, 0]
        d = data_samples["d"][i, :min_len, 0]
        y = out[i, :min_len, 0]

        serle = metrics.erle(np.array(y), np.array(d), np.array(e), segmental=True)
        serle_scores.append(serle)
    return -jnp.mean(jnp.array(serle_scores))


"""
Meta-ID - no extra inputs
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_1024_rl_no_extra_inputs --unroll 16 --extra_signals none --random_roll --outer_loss log_self_mse --dataset linear --true_rir_len 1024

Meta-ID - no acum outer
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_1024_rl_no_acum --unroll 16 --extra_signals udey --random_roll --outer_loss log_indep_mse --dataset linear --true_rir_len 1024

Meta-ID - no log outer
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_1024_rl_no_log --unroll 16 --extra_signals udey --random_roll --outer_loss self_mse --dataset linear --true_rir_len 1024

Meta-ID - no log_acum outer
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_1024_rl_no_log_acum --unroll 16 --extra_signals udey --random_roll --outer_loss indep_mse --dataset linear --true_rir_len 1024

Meta-ID - Unroll 1
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_2_1024_rl --unroll 2 --extra_signals udey --random_roll --outer_loss log_self_mse --dataset linear --true_rir_len 1024

Meta-ID - Unroll 8
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_8_1024_rl --unroll 8 --extra_signals udey --random_roll --outer_loss log_self_mse --dataset linear --true_rir_len 1024

Meta-ID - Unroll 16
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_1024_rl --unroll 16 --extra_signals udey --random_roll --outer_loss log_self_mse --dataset linear --true_rir_len 1024

Meta-ID - Unroll 32
aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_32_1024_rl --unroll 32 --extra_signals udey --random_roll --outer_loss log_self_mse --dataset linear --true_rir_len 1024

Meta-AEC Universal MDF
python aec.py --n_frames 4 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_aec_16_combo_rl_4_1024_512_r2 --unroll 16 --optimizer fgru --random_roll --random_level --outer_loss log_self_mse --double_talk --scene_change --dataset combo --val_loss serle

Meta-AEC AEC Challenge MDF
python aec.py --n_frames 4 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_aec_16_msft_rl_4_1024_512_r2 --unroll 16 --optimizer fgru --random_roll --random_level --outer_loss log_self_mse --double_talk --dataset nonlinear --val_loss serle
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--true_rir_len", type=int, default=None)
    parser.add_argument("--double_talk", action="store_true")
    parser.add_argument("--scene_change", action="store_true")
    parser.add_argument("--random_roll", action="store_true")
    parser.add_argument("--random_level", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--extra_signals", type=str, default="none")
    parser.add_argument("--outer_loss", type=str, default="self_mse")
    parser.add_argument("--val_loss", type=str, default="erle")
    parser.add_argument("--optimizer", type=str, default="gru")
    parser.add_argument("--dataset", type=str, default="linear")
    parser.add_argument("--b1", type=float, default=0.99)

    if parser.parse_known_args()[0].optimizer == "gru":
        parser = optimizer_gru.EGRU.add_args(parser)
        gru_fwd = optimizer_gru._fwd
        gru_grab_args = optimizer_gru.EGRU.grab_args
    elif parser.parse_known_args()[0].optimizer == "fgru":
        parser = optimizer_fgru.FGRU.add_args(parser)
        gru_fwd = optimizer_fgru._fwd
        gru_grab_args = optimizer_fgru.FGRU.grab_args

    parser = AECOLS.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    if kwargs["dataset"] == "linear":
        aec_dataset = MSFTAECDataset_RIR
        dset_kwargs = {
            "double_talk": kwargs["double_talk"],
            "scene_change": kwargs["scene_change"],
            "random_roll": kwargs["random_roll"],
            "random_level": kwargs["random_level"],
            "rir_len": kwargs["true_rir_len"],
        }
    elif kwargs["dataset"] == "nonlinear":
        aec_dataset = MSFTAECDataset
        dset_kwargs = {
            "double_talk": kwargs["double_talk"],
            "scene_change": kwargs["scene_change"],
            "random_roll": kwargs["random_roll"],
            "random_level": kwargs["random_level"],
        }
    elif kwargs["dataset"] == "combo":
        aec_dataset = ComboAECDataset
        dset_kwargs = {
            "random_roll": kwargs["random_roll"],
            "random_level": kwargs["random_level"],
        }
    # make the dataloders
    train_loader = NumpyLoader(
        aec_dataset(
            mode="train",
            max_len=200000,
            **dset_kwargs,
        ),
        batch_size=kwargs["batch_size"],
        shuffle=True,
        persistent_workers=True,
        num_workers=10,
    )
    val_loader = NumpyLoader(
        aec_dataset(
            mode="val",
            **dset_kwargs,
        ),
        batch_size=kwargs["batch_size"],
        persistent_workers=True,
        num_workers=2,
    )
    test_loader = NumpyLoader(
        aec_dataset(
            mode="test",
            **dset_kwargs,
        ),
        batch_size=kwargs["batch_size"],
        num_workers=0,
    )

    # make the callbacks
    callbacks = []
    if not kwargs["debug"]:
        callbacks = [
            CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./meta_ckpts"),
            AudioLoggerCallback(name=kwargs["name"], outputs_base_dir="./meta_outputs"),
            WandBCallback(project="meta-aec", name=kwargs["name"], entity=None),
        ]

    if kwargs["extra_signals"] == "none":
        init_optimizer = optimizer_gru.init_optimizer
        make_mapped_optmizer = optimizer_gru.make_mapped_optmizer
    elif kwargs["extra_signals"] == "udey":
        init_optimizer = optimizer_gru.init_optimizer_all_data
        make_mapped_optmizer = optimizer_gru.make_mapped_optmizer_all_data

    if kwargs["optimizer"] == "fgru":
        gru_fwd = optimizer_fgru._fwd
        init_optimizer = optimizer_fgru.init_optimizer_all_data
        make_mapped_optmizer = optimizer_fgru.make_mapped_optmizer_all_data
        gru_grab_args = optimizer_fgru.FGRU.grab_args
        kwargs["outsize"] = kwargs["n_in_chan"] * kwargs["n_frames"]

    if kwargs["outer_loss"] == "self_mse":
        outer_train_loss = meta_mse_loss
    elif kwargs["outer_loss"] == "indep_mse":
        outer_train_loss = optimizer_utils.frame_indep_meta_mse
    elif kwargs["outer_loss"] == "log_indep_mse":
        outer_train_loss = optimizer_utils.frame_indep_meta_logmse
    elif kwargs["outer_loss"] == "log_self_mse":
        outer_train_loss = meta_log_mse_loss

    if kwargs["val_loss"] == "erle":
        val_loss = neg_erle_val_loss
    elif kwargs["val_loss"] == "serle":
        val_loss = neg_serle_val_loss

    system = MetaAFTrainer(
        _filter_fwd=_AECOLS_fwd,
        filter_kwargs=AECOLS.grab_args(kwargs),
        filter_loss=aec_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=gru_fwd,
        optimizer_kwargs=gru_grab_args(kwargs),
        meta_train_loss=outer_train_loss,
        meta_val_loss=val_loss,
        init_optimizer=init_optimizer,
        make_mapped_optmizer=make_mapped_optmizer,
        callbacks=callbacks,
        kwargs=kwargs,
    )

    # start the training
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(
        **MetaAFTrainer.grab_args(kwargs),
        meta_opt_kwargs={"step_size": 1e-4, "b1": kwargs["b1"]},
        key=key,
    )
