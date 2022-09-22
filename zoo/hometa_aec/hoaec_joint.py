import numpy as np
import argparse
import soundfile as sf
import os
import pandas

import jax
import jax.numpy as jnp
import haiku as hk
from torch.utils.data import Dataset

from metaaf import optimizer_hogru
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf.filter import OverlapSave, OverlapAdd
from metaaf.optimizer_hogru import HOElementWiseGRU, Identity
from metaaf.callbacks import CheckpointCallback, WandBCallback, AudioLoggerCallback

from hometa_aec.hoaec import (
    MSFTAECDataset,
    AECOLS,
    aec_loss,
)

from zoo import metrics

from zoo.__config__ import AEC_DATA_DIR, RES_DATA_DIR


class RESDataset(Dataset):
    def __init__(
        self,
        aec_name,
        mode,
        aec_dir=AEC_DATA_DIR,
        res_dir=RES_DATA_DIR,
        random_roll=False,
        max_len=160000,
    ):

        assert mode in ["train", "val", "test"]

        synthetic_dir = os.path.join(aec_dir, "datasets/synthetic/")

        csv = pandas.read_csv(os.path.join(synthetic_dir, "meta.csv"))
        nearend_scale_dict = {}
        for _, row in csv.iterrows():
            fileid = row["fileid"]
            nearend_scale_dict[fileid] = row["nearend_scale"]
        self.nearend_scale_dict = nearend_scale_dict

        self.farend_speech_dir = os.path.join(synthetic_dir, "farend_speech")
        self.nearend_speech_dir = os.path.join(synthetic_dir, "nearend_speech")
        self.max_len = max_len
        self.random_roll = random_roll
        self.mode = mode

        self.res_dir = os.path.join(res_dir, aec_name)

        np.random.seed(42)
        self.train_random_rolls = np.random.randint(0, 160000, size=9000)

        if self.random_roll:
            if mode == "val":
                np.random.seed(95)
                self.random_rolls = np.random.randint(0, 160000, size=500)
            elif mode == "test":
                np.random.seed(1337)
                self.random_rolls = np.random.randint(0, 160000, size=500)

        """
        if self.mode == "train":
            self.off_set = 0
        elif self.mode == "val":
            self.off_set = 9000
        else:
            self.off_set = 9500
        """

        if self.mode == "test":
            self.offset = 0
        elif self.mode == "val":
            self.offset = 500
        elif self.mode == "train":
            self.offset = 1000

    def __len__(self):
        if self.mode == "train":
            return 9000
        elif self.mode == "val":
            return 500
        else:
            return 500

    def load_from_idx(self, idx):

        idx = idx + self.offset

        # print(idx)

        u, sr = sf.read(
            os.path.join(self.farend_speech_dir, f"farend_speech_fileid_{idx}.wav")
        )

        s, sr = sf.read(
            os.path.join(self.nearend_speech_dir, f"nearend_speech_fileid_{idx}.wav")
        )

        d, _ = sf.read(os.path.join(self.res_dir, f"{idx}_out.wav"))

        if self.mode in ["val", "test"]:
            u = np.pad(u, (0, max(0, self.max_len - len(u))))
            s = np.pad(s, (0, max(0, self.max_len - len(s))))

            if self.random_roll:
                shift = self.random_rolls[idx - self.offset]
                u = np.roll(u, shift)
                s = np.roll(s, shift)

            min_l = min(len(u), len(d), len(s))

            u, d, s = u[:min_l], d[:min_l], s[:min_l]

            u = np.pad(u, (0, max(0, self.max_len - len(u))))
            d = np.pad(d, (0, max(0, self.max_len - len(d))))
            s = np.pad(s, (0, max(0, self.max_len - len(s))))

        else:
            min_l = min(len(u), len(d), len(s))

            u, d, s = u[:min_l], d[:min_l], s[:min_l]

            u = np.pad(u, (0, max(0, self.max_len - len(u))))
            d = np.pad(d, (0, max(0, self.max_len - len(d))))
            s = np.pad(s, (0, max(0, self.max_len - len(s))))

            train_shift = self.train_random_rolls[idx - self.offset]
            u = np.roll(u, train_shift)
            s = np.roll(s, train_shift)

            if self.random_roll:
                shift = np.random.randint(0, self.max_len)
                u = np.roll(u, shift)
                d = np.roll(d, shift)
                s = np.roll(s, shift)

        e = np.zeros(self.max_len)

        s = s * self.nearend_scale_dict[idx]

        return {"d": d[:, None], "u": u[:, None], "e": e[:, None], "s": s[:, None]}

    def __getitem__(self, idx):
        data_dict = self.load_from_idx(idx)

        return {"signals": data_dict, "metadata": {}}


class MaskerGRUOLA(OverlapAdd, hk.Module):
    def __init__(
        self,
        m_window_size,
        m_hop_size,
        m_pad_size,
        m_n_frames,
        m_n_in_chan,
        m_n_out_chan,
        m_is_real=False,
        **kwargs,
    ):
        super().__init__(
            window_size=m_window_size,
            hop_size=m_hop_size,
            pad_size=m_pad_size,
            n_frames=m_n_frames,
            n_in_chan=m_n_in_chan,
            n_out_chan=m_n_out_chan,
            is_real=m_is_real,
            name="Post",
        )

        # select the analysis window
        self.analysis_window = jnp.hanning(self.window_size + 1)[:-1]
        self.synthesis_window = self.get_synthesis_window(self.analysis_window)

        if self.is_real:
            self.out_dim = self.window_size // 2 + 1
        else:
            self.out_dim = self.window_size

        self.h_size = self.out_dim * 2

        self.gru_1 = hk.GRU(self.h_size)
        self.gru_2 = hk.GRU(self.h_size)

        self.lin = hk.Sequential(
            [
                hk.Linear(self.out_dim),
                jax.nn.sigmoid,
            ]
        )

    def get_hidden(self, batch_size):
        hidden_states = hk.get_state(
            "hidden", [2, batch_size, self.h_size], init=hk.initializers.Constant(0)
        )
        return hidden_states

    def update_hidden(self, h_1, h_2):
        return jnp.concatenate((h_1[None, :, :], h_2[None, :, :]), 0)

    def __ola_call__(self, out, u, d, metadata):

        # step 0
        # to correct shape
        batch_size = u.shape[0]
        u = u.reshape((batch_size, -1))
        d = d.reshape((batch_size, -1))

        # step 1
        # extract mic mag and phs
        d_mag = jnp.abs(d)
        d_phs = jnp.exp(1j * jnp.angle(d))

        # step 2
        # extract farend speech mag
        u_mag = jnp.abs(u)

        # step 3
        # combine farend and mic
        ud_feat = (
            jnp.concatenate(
                (jnp.log(d_mag**2 + 10e-10), jnp.log(u_mag**2 + 10e-10)), 1
            )
            / 20.0
        )

        prev_hidden_states = self.get_hidden(batch_size)
        prev_h_1 = prev_hidden_states[0]
        prev_h_2 = prev_hidden_states[1]

        inter_out, h_1 = self.gru_1(ud_feat, prev_h_1)
        gru_out, h_2 = self.gru_2(inter_out + ud_feat, prev_h_2)

        mag_mask = self.lin(gru_out)
        masked = d_mag * mag_mask * d_phs

        new_hidden_states = self.update_hidden(h_1, h_2)
        hk.set_state("hidden", new_hidden_states)

        return masked[:, :, None][0]

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("MaskerDNNOLA")
        parser.add_argument("--m_n_frames", type=int, default=1)
        parser.add_argument("--m_n_in_chan", type=int, default=1)
        parser.add_argument("--m_n_out_chan", type=int, default=1)
        parser.add_argument("--m_window_size", type=int, default=512)
        parser.add_argument("--m_hop_size", type=int, default=256)
        parser.add_argument("--m_pad_size", type=int, default=0)
        parser.add_argument("--m_is_real", action="store_true")
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = [
            "m_n_frames",
            "m_n_in_chan",
            "m_n_out_chan",
            "m_window_size",
            "m_hop_size",
            "m_pad_size",
            "m_is_real",
        ]
        return {k: kwargs[k] for k in keys}


def _gru_ola_fwd(data, out, metadata=None, init_data=None, **kwargs):
    masker = MaskerGRUOLA(**kwargs)
    out["out"] = masker(out=out["out"], u=data["u"], d=data["d"])
    return out


class AECIdentity(OverlapSave, hk.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # select the analysis window
        self.analysis_window = jnp.ones(self.window_size)

    def __ols_call__(self, u, metadata):
        w = self.get_filter(name="w")

        return {
            "out": u,
            "u": u,
            "loss": 0.0,
        }

    @staticmethod
    def add_args(parent_parser):
        return super(AECIdentity, AECIdentity).add_args(parent_parser)

    @staticmethod
    def grab_args(kwargs):
        return super(AECIdentity, AECIdentity).grab_args(kwargs)


def _AECIdentity_fwd(u, d, e, s, metadata=None, init_data=None, **kwargs):
    optimizee = AECIdentity(**kwargs)
    return optimizee(u=u)


def simple_stft(x, window_size, hop_size):
    x = jnp.pad(x, ((0, window_size), (0, 0)))
    n_frames = (len(x) - window_size) // hop_size
    window_idx = jnp.arange(window_size)[None, :]
    frame_idx = jnp.arange(n_frames)[:, None]
    window_idxs = window_idx + frame_idx * hop_size

    # index the buffer with the map and window
    analysis_window = jnp.hanning(window_size + 1)[:-1]
    windowed_x = x[window_idxs] * analysis_window[None, :, None]

    # 0 is T, 1 will be F
    stft_x = jnp.fft.rfft(windowed_x, axis=1) / jnp.sqrt(window_size)
    return stft_x


def make_meta_sup_mse_loss(window_size, hop_size, time_domain, freq_domain):
    buffer_size = window_size - hop_size
    assert time_domain or freq_domain

    def outer_meta_mse_loss(losses, outputs, data_samples, metadata, outer_learnable):
        out = jnp.concatenate(outputs["out"], 0)

        print("check out shape")
        print(out.shape)

        s_hat = out[buffer_size:]
        s = data_samples["s"][: len(s_hat)]

        if time_domain:
            td_loss = jnp.mean(jnp.abs(s - s_hat) ** 2)
        else:
            td_loss = 0

        if freq_domain:
            S = jnp.abs(simple_stft(s, window_size, hop_size))
            S_hat = jnp.abs(simple_stft(s_hat, window_size, hop_size))
            mag_loss = jnp.mean(jnp.abs(S - S_hat) ** 2)
        else:
            mag_loss = 0

        total_loss = td_loss + mag_loss

        return total_loss / (int(time_domain) + int(freq_domain))

    return outer_meta_mse_loss


def make_neg_sisdr_val(window_size, hop_size):
    buffer_size = window_size - hop_size

    def neg_sisdr_val(losses, outputs, data_samples, metadata, outer_learnable):
        sisdr_scores = []

        out = outputs["out"]
        out = out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[-1])

        for i in range(len(out)):
            out_trim = out[:, buffer_size:, :]

            min_len = min(out_trim.shape[1], data_samples["s"].shape[1])

            clean = data_samples["s"][i, :min_len, 0]
            enhanced = out_trim[i, :min_len, 0]

            sisdr_scores.append(metrics.sisdr(enhanced, clean))

        return -jnp.mean(jnp.array(sisdr_scores))

    return neg_sisdr_val


if __name__ == "__main__":

    """
    - joint mode:
        - res: train dnn-res to do echo&noise suppression without aec
            - denoising: train dnn-res to do noise suppression/upper bound
        - aec-res:
            - train: train dnn-res to do res on outputs of autodsp, need aec name
            - joint_train: init meta-aec + dnn-res randomly and train from scratch, not supported

    """

    import pprint

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--double_talk", action="store_true")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--random_roll", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--joint_mode", type=str, default="res")
    parser.add_argument("--aec_res_mode", type=str, default="")
    parser.add_argument("--aec_name", type=str, default="")
    parser.add_argument("--denoising", action="store_true")

    parser = AECOLS.add_args(parser)
    parser = HOElementWiseGRU.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    parser = MaskerGRUOLA.add_args(parser)

    kwargs = vars(parser.parse_args())

    assert kwargs["joint_mode"] in ["res", "aec_res"]

    if kwargs["joint_mode"] == "aec_res":
        assert kwargs["aec_res_mode"] == "train"
    else:
        assert kwargs["aec_res_mode"] == ""

    # RFFT
    kwargs["freq_size"] = kwargs["window_size"] // 2 + 1  # K in paper

    if kwargs["group_mode"] == "banded":
        # overlapping
        kwargs["c_size"] = int(
            np.ceil(kwargs["freq_size"] / (kwargs["group_size"] // 2))
        )
    else:
        # no overlapping
        kwargs["c_size"] = int(np.ceil(kwargs["freq_size"] / kwargs["group_size"]))

    pprint.pprint(kwargs)

    # make functions
    meta_sup_mse_loss = make_meta_sup_mse_loss(
        kwargs["m_window_size"], kwargs["m_hop_size"], False, True
    )
    neg_sisdr_val = make_neg_sisdr_val(kwargs["m_window_size"], kwargs["m_hop_size"])
    _postprocess_fwd = _gru_ola_fwd

    # make the dataloders
    if kwargs["joint_mode"] == "res":
        train_loader = NumpyLoader(
            MSFTAECDataset(
                mode="train",
                double_talk=kwargs["double_talk"],
                random_roll=kwargs["random_roll"],
                denoising=kwargs["denoising"],
            ),
            batch_size=kwargs["batch_size"],
            shuffle=True,
            num_workers=10,
        )
        val_loader = NumpyLoader(
            MSFTAECDataset(
                mode="val",
                double_talk=kwargs["double_talk"],
                random_roll=kwargs["random_roll"],
                denoising=kwargs["denoising"],
            ),
            batch_size=kwargs["batch_size"],
            num_workers=2,
        )
        test_loader = NumpyLoader(
            MSFTAECDataset(
                mode="test",
                double_talk=kwargs["double_talk"],
                random_roll=kwargs["random_roll"],
                denoising=kwargs["denoising"],
            ),
            batch_size=kwargs["batch_size"],
            num_workers=0,
        )

    else:
        aec_name = kwargs["aec_name"]

        train_loader = NumpyLoader(
            RESDataset(aec_name=aec_name, mode="train"),
            batch_size=kwargs["batch_size"],
            shuffle=True,
            num_workers=10,
        )
        val_loader = NumpyLoader(
            RESDataset(
                aec_name=aec_name, mode="val", random_roll=kwargs["random_roll"]
            ),
            batch_size=kwargs["batch_size"],
            num_workers=2,
        )
        test_loader = NumpyLoader(
            RESDataset(
                aec_name=aec_name, mode="test", random_roll=kwargs["random_roll"]
            ),
            batch_size=kwargs["batch_size"],
            num_workers=2,
        )

    # make the callbacks
    callbacks = [
        CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./ckpts"),
        AudioLoggerCallback(name=kwargs["name"], outputs_base_dir="./outputs"),
        WandBCallback(project="higher_order_aec", name=kwargs["name"], entity=None),
    ]

    # setup some things for the optimizer
    system = MetaAFTrainer(
        _filter_fwd=_AECIdentity_fwd,
        filter_kwargs=AECIdentity.grab_args(kwargs),
        filter_loss=aec_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        meta_train_loss=meta_sup_mse_loss,
        meta_val_loss=neg_sisdr_val,
        _optimizer_fwd=optimizer_hogru._identity_fwd,
        optimizer_kwargs=Identity.grab_args(kwargs),
        init_optimizer=optimizer_hogru.init_optimizer_identity,
        make_mapped_optmizer=optimizer_hogru.make_mapped_optmizer_identity,
        _postprocess_fwd=_postprocess_fwd,
        postprocess_kwargs=MaskerGRUOLA.grab_args(kwargs),
        callbacks=callbacks,
        kwargs=kwargs,
    )

    # start the training
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(
        **MetaAFTrainer.grab_args(kwargs),
        meta_opt_kwargs={"step_size": kwargs["lr"]},
        key=key,
    )
