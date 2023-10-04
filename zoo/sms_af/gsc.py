import argparse
import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf.optimizer_utils import clip_grads
from metaaf.filter import make_inner_passthrough
from metaaf import optimizer_hofgru_simple
from metaaf import preprocess_utils
from metaaf import postprocess_utils

from metaaf.callbacks import CheckpointCallback, WandBCallback
from zoo import metrics
from zoo.gsc.gsc import (
    Chime3Dataset,
    GSCOLA,
    _GSCOLA_fwd,
    gsc_loss,
    meta_log_mse_loss,
)


def simple_stft(x):
    window_size = 512
    hop_size = 256
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


def make_log_mse_loss(window_size, hop_size):
    buffer_size = window_size - hop_size
    EPS = 1e-9

    def log_mse_loss(
        losses, outputs, data_samples, metadata, outer_index, outer_learnable
    ):
        s_hat = jnp.concatenate(outputs["out"], 0)[buffer_size:, 0, None]
        s = data_samples["s"][: len(s_hat), 0, None]
        return jnp.log(jnp.mean(jnp.abs(s - s_hat) ** 2) + EPS)

    return log_mse_loss


def make_sisdr_loss(window_size, hop_size):
    buffer_size = window_size - hop_size
    EPS = 1e-10

    def neg_sisdr_loss(
        losses, outputs, data_samples, metadata, outer_index, outer_learnable
    ):
        s_hat = jnp.concatenate(outputs["out"], 0)[buffer_size:, 0, None]
        s = data_samples["s"][: len(s_hat), 0, None]

        s_proj = (s * s_hat).mean() / (s**2 + EPS).mean() * s
        return -10 * jnp.log10(
            (s_hat**2).mean() / ((s_hat - s_proj) ** 2 + EPS).mean() + EPS
        )

    return neg_sisdr_loss


def make_neg_sisdr_val(window_size, hop_size):
    buffer_size = window_size - hop_size

    def neg_sisdr_val(losses, outputs, data_samples, metadata, outer_learnable):
        out = jnp.reshape(
            outputs["out"],
            (outputs["out"].shape[0], -1, outputs["out"].shape[-1]),
        )

        sisdr_scores = []
        for i in range(len(out)):
            out_trim = out[i, buffer_size:]
            min_len = min(out_trim.shape[0], data_samples["s"].shape[1])

            clean = data_samples["s"][i, :min_len, 0]
            enhanced = out_trim[:min_len, 0]

            sisdr_scores.append(metrics.sisdr(enhanced, clean))
        return -jnp.mean(jnp.array(sisdr_scores))

    return neg_sisdr_val


def get_loss(kwargs):
    if kwargs["loss"] == "log_mse":
        outer_loss = make_log_mse_loss(kwargs["window_size"], kwargs["hop_size"])
    elif kwargs["loss"] == "self_log_mse":
        outer_loss = meta_log_mse_loss
    elif kwargs["loss"] == "sisdr":
        outer_loss = make_sisdr_loss(kwargs["window_size"], kwargs["hop_size"])
    return outer_loss


def get_val(kwargs):
    if kwargs["val_metric"] == "neg_sisdr":
        val_metric = make_neg_sisdr_val(kwargs["window_size"], kwargs["hop_size"])
    return val_metric


"""
Run this command to train a PUPU supevised gsc model.

python gsc.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --n_devices 1 --batch_size 16 --group_size 5 --group_hop 2 --h_size 64 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 3 --name gsc_sisdr_val_sisdr_2iter_posterior --unroll 128 --cov_init identity --cov_update oracle --exp_avg .9 --steer_method rank1 --cov_update_regularizer 0.01 --loss sisdr --val_metric neg_sisdr --lr 5e-4 --inner_iterations 2 --auto_posterior --features_to_use uef --debug

Remove the --auto_posterior flag to train a -P model and increase the --inner_iterations argument to train a PUPUx2 model. Model size can be changed via the --h_size argument and supervision can be changes by switching between the sisdr and self_log_mse losses in the --loss command.

"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="neg_sisdr")
    parser.add_argument("--val_metric", type=str, default="sisdr")
    parser.add_argument("--debug", action="store_true")

    parser = optimizer_hofgru_simple.HO_FGRU.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    parser = GSCOLA.add_args(parser)
    kwargs = vars(parser.parse_args())
    kwargs["outsize"] = kwargs["n_in_chan"] - 1
    pprint.pprint(kwargs)

    dset = Partial(Chime3Dataset, n_mics=kwargs["n_in_chan"])

    # make the dataloders
    train_loader = NumpyLoader(
        dset(
            mode="train",
            signal_len=128000,
        ),
        batch_size=kwargs["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
    )

    val_loader = NumpyLoader(
        dset(
            mode="val",
            signal_len=128000,
        ),
        batch_size=kwargs["batch_size"],
        drop_last=True,
        num_workers=2,
        persistent_workers=True,
    )

    test_loader = NumpyLoader(
        dset(
            mode="test",
            signal_len=128000,
        ),
        batch_size=1,
        num_workers=0,
    )

    # make the callbacks
    callbacks = []
    if not kwargs["debug"]:
        callbacks = [
            CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./ckpts/gsc"),
            WandBCallback(project="sms-af", name=kwargs["name"], entity=None),
        ]

    system = MetaAFTrainer(
        _filter_fwd=_GSCOLA_fwd,
        filter_kwargs=GSCOLA.grab_args(kwargs),
        filter_loss=gsc_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=optimizer_hofgru_simple._fwd,
        optimizer_kwargs=optimizer_hofgru_simple.HO_FGRU.grab_args(kwargs),
        meta_train_loss=get_loss(kwargs),
        meta_val_loss=get_val(kwargs),
        init_optimizer=optimizer_hofgru_simple.init_optimizer_all_data,
        make_mapped_optimizer=optimizer_hofgru_simple.make_mapped_optimizer_all_data,
        make_train_mapped_optimizer=optimizer_hofgru_simple.make_train_mapped_optimizer_all_data,
        make_get_filter_features=make_inner_passthrough,
        _preprocess_fwd=preprocess_utils._identity_fwd,
        preprocess_kwargs={},
        _postprocess_fwd=postprocess_utils._identity_fwd,
        postprocess_kwargs={},
        inner_iterations=kwargs["inner_iterations"],
        auto_posterior=kwargs["auto_posterior"],
        callbacks=callbacks,
        kwargs=kwargs,
    )

    # start the training
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(
        **MetaAFTrainer.grab_args(kwargs),
        meta_opt_kwargs={"step_size": kwargs["lr"], "b1": 0.9},
        meta_opt_preprocess=Partial(clip_grads, max_norm=5, eps=1e-9),
        key=key,
    )
