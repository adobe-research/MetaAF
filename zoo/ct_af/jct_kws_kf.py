import argparse
import pickle
import os

import jax
from jax.tree_util import Partial

from metaaf.optimizer_utils import clip_grads
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer

from metaaf.callbacks import CheckpointCallback, WandBCallback

from train_kws import KWSAECDataset, load_kws_model
from zoo.aec.aec import AECOLS, _AECOLS_fwd, aec_loss
from ct_metaaec_kws import (
    make_meta_classification_loss,
    make_meta_classification_val,
)
import optimizer_kf as kf


"""
python jct_kws_kf.py --n_frames 4 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 16 --total_epochs 1000 --val_period 5 --reduce_lr_patience 1 --early_stop_patience 5 --name 1_kws_loss_kf_25_0_tweaked --unroll 93 --kws_loc ./ckpts/35cmds_tcn.pkl --kws_mode 35cmds --lr 1e-4 --max_norm 1 --b1 0.9 --kf_ckpt kf_35_25_0 2023_03_28_21_51_51 0
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--kws_loc", type=str, default="")
    parser.add_argument("--kws_mode", type=str, default="10cmds")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--b1", type=float, default=0.99)
    parser.add_argument("--max_norm", type=float, default=10)
    parser.add_argument(
        "--kf_ckpt",
        nargs="+",
        default=["kf_35_25_0", "2023_03_28_21_51_51", "0"],
    )

    parser = kf.add_args(parser)
    parser = AECOLS.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    # make the callbacks
    callbacks = []
    if not kwargs["debug"]:
        callbacks = [
            CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./kws_ckpts"),
            WandBCallback(project="ct-metaaec", name=kwargs["name"], entity=None),
        ]

    outer_train_loss = make_meta_classification_loss(
        kwargs["kws_loc"],
        1.0,
        kwargs,
        joint_train_kws=True,
        no_aec=False,
        joint_train_aec=False,
    )

    dset = KWSAECDataset
    train_loader = NumpyLoader(
        dset(max_len=64000, mode="train", kws_mode=kwargs["kws_mode"]),
        batch_size=kwargs["batch_size"],
        num_workers=6,
        persistent_workers=True,
        shuffle=True,
    )
    val_loader = NumpyLoader(
        dset(max_len=64000, mode="val", kws_mode=kwargs["kws_mode"]),
        num_workers=2,
        batch_size=kwargs["batch_size"],
    )
    test_loader = NumpyLoader(
        dset(max_len=64000, mode="test", kws_mode=kwargs["kws_mode"]),
        batch_size=kwargs["batch_size"],
    )

    system = MetaAFTrainer(
        _filter_fwd=_AECOLS_fwd,
        filter_kwargs=AECOLS.grab_args(kwargs),
        filter_loss=aec_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=kf._fwd,
        optimizer_kwargs=kf.grab_args(kwargs),
        meta_train_loss=outer_train_loss,
        meta_val_loss=make_meta_classification_val(
            kwargs["kws_loc"], joint_train_kws=True
        ),
        init_optimizer=kf.init_optimizer,
        make_mapped_optimizer=kf.make_mapped_optimizer,
        callbacks=callbacks,
        kwargs=kwargs,
    )

    outer_learnable = None
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    outer_learnable = system.init_outer_learnable(subkey)

    # load a pre-trained KWS module
    kws_p, _ = load_kws_model(kwargs["kws_loc"])
    outer_learnable["kws_p"] = kws_p

    # load a pre-trained AEC module
    e = kwargs["kf_ckpt"][2]
    ckpt_loc = os.path.join(
        "./kws_ckpts",
        kwargs["kf_ckpt"][0],
        kwargs["kf_ckpt"][1],
        f"epoch_{e}.pkl",
    )
    with open(ckpt_loc, "rb") as f:
        init_ckpt = pickle.load(f)

    for k in init_ckpt:
        outer_learnable[k] = init_ckpt[k]

    # start the training
    outer_learned, losses = system.train(
        **MetaAFTrainer.grab_args(kwargs),
        meta_opt_kwargs={"step_size": kwargs["lr"], "b1": kwargs["b1"]},
        meta_opt_preprocess=Partial(clip_grads, max_norm=kwargs["max_norm"], eps=1e-9),
        outer_learnable=outer_learnable,
        key=key,
    )
