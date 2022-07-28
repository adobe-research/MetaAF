import numpy as np
import scipy
import argparse
import soundfile as sf
import os
import glob2

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
from metaaf.callbacks import CheckpointCallback, WandBCallback, AudioLoggerCallback


from zoo import metrics
from zoo.__config__ import AEC_DATA_DIR, RIR_DATA_DIR


class MSFTAECDataset(Dataset):
    def __init__(
        self,
        base_dir=AEC_DATA_DIR,
        mode="train",
        double_talk=True,
        scene_change=False,
        random_roll=False,
        max_len=160000,
    ):

        synthetic_dir = os.path.join(base_dir, "datasets/synthetic/")

        self.echo_signal_dir = os.path.join(synthetic_dir, "echo_signal")
        self.farend_speech_dir = os.path.join(synthetic_dir, "farend_speech")
        self.nearend_mic_dir = os.path.join(synthetic_dir, "nearend_mic_signal")
        self.nearend_speech_dir = os.path.join(synthetic_dir, "nearend_speech")
        self.max_len = max_len

        self.double_talk = double_talk
        self.scene_change = scene_change
        self.random_roll = random_roll
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
            if mode == "val":
                np.random.seed(95)
                self.random_rolls = np.random.randint(0, 160000, size=500)
            elif mode == "test":
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

        e, sr = sf.read(os.path.join(self.echo_signal_dir, f"echo_fileid_{idx}.wav"))
        s, sr = sf.read(
            os.path.join(self.nearend_speech_dir, f"nearend_speech_fileid_{idx}.wav")
        )

        u = np.pad(u, (0, max(0, self.max_len - len(u))))
        d = np.pad(d, (0, max(0, self.max_len - len(d))))
        e = np.pad(e, (0, max(0, self.max_len - len(e))))
        s = np.pad(s, (0, max(0, self.max_len - len(s))))

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


class MSFTAECDataset_RIR(Dataset):
    def __init__(
        self,
        aec_dir=AEC_DATA_DIR,
        rir_dir=RIR_DATA_DIR,
        mode="train",
        double_talk=True,
        scene_change=False,
        random_roll=False,
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
        self.max_len = max_len
        self.rir_len = rir_len

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
                np.abs(e ** 2).mean() * (10 ** (ser / 10)) / np.abs(s ** 2).mean()
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
                start = 64000  # int(0.4 * self.max_len)
                stop = 96000  # int(0.6 * self.max_len)
                change_idx = (
                    np.random.randint(start, stop)
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

    def __ols_call__(self, u, d, metadata):
        w = self.get_filter(name="w")

        d_hat = (w * u).sum(0)
        out = d[-1] - d_hat
        return {
            "out": out,
            "u": u,
            "d": d[-1, None],
            "e": out[None],
            "loss": jnp.vdot(out, out).real / out.size,
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


def aec_loss(out, data_samples, metadata):
    return out["loss"]


def meta_mse_loss(losses, outputs, data_samples, metadata, outer_learnable):
    return jnp.mean(jnp.abs(outputs) ** 2)


def meta_log_mse_loss(losses, outputs, data_samples, metadata, outer_learnable):
    EPS = 1e-8
    return jnp.log(jnp.mean(jnp.abs(outputs) ** 2) + EPS)


def neg_erle_val_loss(losses, outputs, data_samples, metadata, outer_learnable):
    erle_scores = []
    for i in range(len(outputs)):
        min_len = min(
            outputs.shape[1], data_samples["d"].shape[1], data_samples["e"].shape[1]
        )

        e = data_samples["e"][i, :min_len, 0]
        d = data_samples["d"][i, :min_len, 0]
        y = outputs[i, :min_len, 0]

        erle = metrics.erle(np.array(y), np.array(d), np.array(e))
        erle_scores.append(erle)
    return -jnp.mean(jnp.array(erle_scores))


"""
Initial Ablation
Meta-ID
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_c --unroll 16 --extra_signals ude --random_roll --outer_loss log_self_mse

Meta-ID - extra inputs
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_no_inputs_c --unroll 16 --extra_signals none --random_roll --outer_loss log_self_mse

Meta-ID - acum outer
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_no_acum_wlog_c --unroll 16 --extra_signals ude --random_roll --outer_loss log_indep_mse

Meta-ID - log outer
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_no_log_c --unroll 16 --extra_signals ude --random_roll --outer_loss self_mse

Meta-ID - log_acum outer
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_16_no_acum_c --unroll 16 --extra_signals ude --random_roll --outer_loss indep_mse

Meta-ID - unroll = 1 (set to 2)
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_1_c --unroll 2 --extra_signals ude --random_roll --outer_loss log_self_mse

Meta-ID - unroll = 8
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_8_c --unroll 8 --extra_signals ude --random_roll --outer_loss log_self_mse

Meta-ID - unroll = 32
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_id_32_c --unroll 32 --extra_signals ude --random_roll --outer_loss log_self_mse

AEC Experiments
Meta-AEC Double Talk
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_aec_16_dt_c --unroll 16 --extra_signals ude --random_roll --outer_loss log_self_mse --double_talk

Meta-AEC Double Talk Scene Change
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_aec_16_dt_sc_c --unroll 16 --extra_signals ude --random_roll --outer_loss log_self_mse --double_talk --scene_change

Meta-AEC Double Talk Nonlinear
python aec.py --n_frames 1 --window_size 2048 --hop_size 1024 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 2 --batch_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 4 --name meta_aec_16_dt_nl_c --unroll 16 --extra_signals ude --random_roll --outer_loss log_self_mse --double_talk --dataset nonlinear
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--double_talk", action="store_true")
    parser.add_argument("--scene_change", action="store_true")
    parser.add_argument("--random_roll", action="store_true")

    parser.add_argument("--extra_signals", type=str, default="none")
    parser.add_argument("--outer_loss", type=str, default="self_mse")
    parser.add_argument("--dataset", type=str, default="linear")
    parser.add_argument("--b1", type=float, default=0.99)

    parser = optimizer_gru.ElementWiseGRU.add_args(parser)
    parser = AECOLS.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    if kwargs["dataset"] == "linear":
        aec_dataset = MSFTAECDataset_RIR
    elif kwargs["dataset"] == "nonlinear":
        aec_dataset = MSFTAECDataset

    # make the dataloders
    train_loader = NumpyLoader(
        aec_dataset(
            mode="train",
            double_talk=kwargs["double_talk"],
            scene_change=kwargs["scene_change"],
            random_roll=kwargs["random_roll"],
            max_len=200000,
        ),
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=8,
    )
    val_loader = NumpyLoader(
        aec_dataset(
            mode="val",
            double_talk=kwargs["double_talk"],
            scene_change=kwargs["scene_change"],
            random_roll=kwargs["random_roll"],
        ),
        batch_size=kwargs["batch_size"],
        num_workers=4,
    )
    test_loader = NumpyLoader(
        aec_dataset(
            mode="test",
            double_talk=kwargs["double_talk"],
            scene_change=kwargs["scene_change"],
            random_roll=kwargs["random_roll"],
        ),
        batch_size=kwargs["batch_size"],
        num_workers=0,
    )

    # make the callbacks
    callbacks = [
        CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./taslp_ckpts"),
        AudioLoggerCallback(name=kwargs["name"], outputs_base_dir="./taslp_outputs"),
        WandBCallback(project="meta-aec", name=kwargs["name"], entity=None),
    ]

    if kwargs["extra_signals"] == "none":
        init_optimizer = optimizer_gru.init_optimizer
        make_mapped_optmizer = optimizer_gru.make_mapped_optmizer
    elif kwargs["extra_signals"] == "ude":
        init_optimizer = optimizer_gru.init_optimizer_all_data
        make_mapped_optmizer = optimizer_gru.make_mapped_optmizer_all_data

    if kwargs["outer_loss"] == "self_mse":
        outer_train_loss = meta_mse_loss
    elif kwargs["outer_loss"] == "indep_mse":
        outer_train_loss = optimizer_utils.frame_indep_meta_mse
    elif kwargs["outer_loss"] == "log_indep_mse":
        outer_train_loss = optimizer_utils.frame_indep_meta_logmse
    elif kwargs["outer_loss"] == "log_self_mse":
        outer_train_loss = meta_log_mse_loss

    system = MetaAFTrainer(
        _filter_fwd=_AECOLS_fwd,
        filter_kwargs=AECOLS.grab_args(kwargs),
        filter_loss=aec_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=optimizer_gru._elementwise_gru_fwd,
        optimizer_kwargs=optimizer_gru.ElementWiseGRU.grab_args(kwargs),
        meta_train_loss=outer_train_loss,
        meta_val_loss=neg_erle_val_loss,
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
