import argparse
import jax
import haiku as hk
import jax.numpy as jnp
from jax.tree_util import Partial
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_hofgru_simple
from metaaf.meta_optimizers import complex_adam
from metaaf.optimizer_utils import clip_grads
from metaaf import postprocess_utils

from metaaf.callbacks import CheckpointCallback, WandBCallback
from metaaf.filter import (
    OverlapAdd,
    make_inner_passthrough,
    make_inner_grad,
)

from zoo.aec.aec import (
    meta_log_mse_loss,
    aec_loss,
    MSFTAECDataset,
)
from zoo import metrics


class AECOLA(OverlapAdd, hk.Module):
    def __init__(self, antialias_outputs, no_analysis_window, **kwargs):
        super().__init__(**kwargs)
        self.no_analysis_window = no_analysis_window
        self.antialias_outputs = antialias_outputs

        # select the analysis window
        if self.no_analysis_window:
            self.analysis_window = jnp.ones(self.window_size) * (
                2 * self.hop_size / self.window_size
            )
            self.synthesis_window = jnp.hanning(self.window_size + 1)[:-1]
        else:
            self.analysis_window = jnp.hanning(self.window_size + 1)[:-1] ** 0.5
            self.synthesis_window = self.get_synthesis_window(self.analysis_window)

    def antialias(self, x):
        return self.fft(
            self.ifft(x, axis=1) * self.synthesis_window[None, :, None], axis=1
        )

    def __ola_call__(self, u, d, metadata):
        w = self.get_filter(name="w")
        y = (w * u).sum(0)

        out = d[-1] - y

        if self.antialias_outputs:
            y = self.antialias(y[None])
            e = self.antialias(out[None])
        else:
            y = y[None]
            e = out[None]

        return {
            "out": out,
            "u": u,
            "d": d[-1, None],
            "e": e,
            "y": y,
            "w": w,
            "grad": jnp.conj(u) * e,
            "loss": jnp.vdot(e, e).real / 2,
        }

    @staticmethod
    def add_args(parent_parser):
        parent_parser = super(AECOLA, AECOLA).add_args(parent_parser)
        parser = parent_parser.add_argument_group("AECOLA")
        parser.add_argument("--antialias_outputs", action="store_true")
        parser.add_argument("--no_analysis_window", action="store_true")
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        class_keys = {}
        if "antialias_outputs" in kwargs:
            class_keys.update({"antialias_outputs": kwargs["antialias_outputs"]})
        else:
            class_keys["antialias_outputs"] = True

        if "no_analysis_window" in kwargs:
            class_keys.update({"no_analysis_window": kwargs["no_analysis_window"]})
        else:
            class_keys["no_analysis_window"] = True
        class_keys.update(super(AECOLA, AECOLA).grab_args(kwargs))
        return class_keys


def _AECOLA_fwd(u, d, e, s, metadata=None, init_data=None, **kwargs):
    gen_filter = AECOLA(**kwargs)
    return gen_filter(u=u, d=d)


def make_neg_sisdr_val(hop_size):
    def neg_sisdr_val(losses, outputs, data_samples, metadata, outer_learnable):
        out = jnp.reshape(
            outputs["out"],
            (outputs["out"].shape[0], -1, outputs["out"].shape[-1]),
        )
        out = out[:, hop_size:, :]

        sisdr_scores = []
        for i in range(len(out)):
            out_trim = out[i]
            min_len = min(out_trim.shape[0], data_samples["s"].shape[1])

            clean = data_samples["s"][i, :min_len, 0]
            enhanced = out_trim[:min_len, 0]

            sisdr_scores.append(metrics.sisdr(enhanced, clean))
        return -jnp.mean(jnp.array(sisdr_scores))

    return neg_sisdr_val


def make_sup_log_echo_td_loss(buffer_size):
    def sup_log_echo_td_loss_ola(
        losses, outputs, data_samples, metadata, outer_index, outer_learnable
    ):
        d_hat = jnp.concatenate(outputs["out"], 0)[buffer_size:]
        e_hat = data_samples["d"][: len(d_hat)] - d_hat
        e = data_samples["e"][: len(e_hat)]
        return jnp.log(jnp.mean(jnp.abs(e - e_hat) ** 2) + 1e-10)

    return sup_log_echo_td_loss_ola


def get_val_loss(kwargs):
    val_loss = None
    if kwargs["val_loss"] == "neg_sisdr_ola":
        val_loss = make_neg_sisdr_val(kwargs["window_size"] - kwargs["hop_size"])
    return val_loss


def get_loss(kwargs):
    outer_loss = None
    if kwargs["loss"] == "sup_echo_td_ola":
        outer_loss = make_sup_log_echo_td_loss(
            kwargs["window_size"] - kwargs["hop_size"]
        )
    else:
        outer_loss = meta_log_mse_loss

    return outer_loss


"""
Run this command to train a PU supevised AEC model.

python aec.py --n_frames 8 --window_size 512 --hop_size 256 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 16 --group_size 5 --group_hop 2 --h_size 32 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 3 --unroll 128 --name sms_aec_m_s_pu --features_to_use uef --no_inner_grad --loss sup_echo_td_ola --val_loss neg_sisdr_ola --inner_iterations 1 --auto_posterior --no_analysis_window --debug

Remove the --auto_posterior flag to train a -P model and increase the --inner_iterations argument to train a PUPUx2 model. Model size can be changed via the --h_size argument and supervision can be changes by switching between the sup_echo_td_ola and unsupervised losses in the --loss command.
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--max_norm", type=float, default=5)
    parser.add_argument("--loss", type=str, default="stoi")
    parser.add_argument("--val_loss", type=str, default="neg_stoi")
    parser.add_argument("--real_data", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts/aec")
    parser.add_argument("--debug", action="store_true")

    optimizer_pkg = optimizer_hofgru_simple
    _filt_fwd = _AECOLA_fwd
    filter_grab_args = AECOLA.grab_args
    filter_add_args = AECOLA.add_args

    parser = optimizer_pkg.HO_FGRU.add_args(parser)
    parser = filter_add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    kwargs["outsize"] = kwargs["n_frames"]

    pprint.pprint(kwargs)

    train_aec_dataset = MSFTAECDataset(
        mode="train",
        double_talk=True,
        random_roll=True,
        random_level=True,
        max_len=160000,
        audio_reader="soundfile",
    )
    val_aec_dataset = MSFTAECDataset(
        mode="val",
        double_talk=True,
        random_roll=True,
        random_level=True,
        max_len=160000,
        audio_reader="soundfile",
    )
    test_aec_dataset = MSFTAECDataset(
        mode="train",
        double_talk=True,
        random_roll=True,
        random_level=True,
        max_len=160000,
        audio_reader="soundfile",
    )

    # make the dataloders
    train_loader = NumpyLoader(
        train_aec_dataset,
        batch_size=kwargs["batch_size"],
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
    )
    val_loader = NumpyLoader(
        val_aec_dataset,
        batch_size=kwargs["batch_size"],
        num_workers=2,
    )
    test_loader = NumpyLoader(
        test_aec_dataset,
        batch_size=kwargs["batch_size"],
        num_workers=0,
    )

    # make the callbacks
    callbacks = []
    if not kwargs["debug"]:
        callbacks = [
            CheckpointCallback(name=kwargs["name"], ckpt_base_dir=kwargs["ckpt_dir"]),
            WandBCallback(project="sms-af", name=kwargs["name"], entity=None),
        ]

    system = MetaAFTrainer(
        _filter_fwd=_filt_fwd,
        filter_kwargs=filter_grab_args(kwargs),
        filter_loss=aec_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=optimizer_pkg._fwd,
        optimizer_kwargs=optimizer_pkg.HO_FGRU.grab_args(kwargs),
        meta_train_loss=get_loss(kwargs),
        meta_val_loss=get_val_loss(kwargs),
        init_optimizer=optimizer_pkg.init_optimizer_all_data,
        make_mapped_optimizer=optimizer_pkg.make_mapped_optimizer_all_data,
        make_train_mapped_optimizer=optimizer_pkg.make_train_mapped_optimizer_all_data,
        make_get_filter_features=make_inner_passthrough
        if kwargs["no_inner_grad"]
        else make_inner_grad,
        _postprocess_fwd=postprocess_utils._identity_fwd,
        postprocess_kwargs={},
        callbacks=callbacks,
        kwargs=kwargs,
    )

    #  load a pre-trained AEC module
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    outer_learnable = system.init_outer_learnable(subkey)

    # start the training
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(
        **MetaAFTrainer.grab_args(kwargs),
        meta_opt=complex_adam,
        meta_opt_kwargs={"step_size": kwargs["lr"], "b1": kwargs["b1"]},
        meta_opt_preprocess=Partial(clip_grads, max_norm=kwargs["max_norm"], eps=1e-9),
        outer_learnable=outer_learnable,
        key=key,
    )
