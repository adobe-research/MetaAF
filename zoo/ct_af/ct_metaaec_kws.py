import numpy as np
import argparse
import pickle
import os

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_hofgru_aug

from metaaf.callbacks import CheckpointCallback, WandBCallback
from metaaf.optimizer_utils import clip_grads
from train_kws import RealKWSAECDataset, KWSAECDataset, load_kws_model
from zoo.aec.aec import AECOLS, _AECOLS_fwd, aec_loss


def make_meta_classification_loss(
    loc,
    alpha,
    kwargs,
    joint_train_kws=False,
    no_aec=False,
    joint_train_aec=True,
):
    kws_params, kws_apply = load_kws_model(loc)

    def meta_classification_loss(
        losses, outputs, data_samples, metadata, outer_index, outer_learnable
    ):
        if joint_train_kws:
            cur_kws_params = outer_learnable["kws_p"]
        else:
            cur_kws_params = kws_params

        kws_input = jnp.concatenate(outputs["out"], 0)

        aec_loss = jnp.log(jnp.mean(jnp.abs(kws_input) ** 2) + 1e-8)

        if not joint_train_aec:
            kws_input = jax.lax.stop_gradient(kws_input)

        if no_aec:
            kws_input = data_samples["d"]

        preds = kws_apply(cur_kws_params, None, kws_input)
        targets = metadata["onehot"]
        kws_loss = -np.mean(np.sum(jnp.log(preds) * targets, axis=-1))

        return alpha * kws_loss + (1.0 - alpha) * aec_loss

    return meta_classification_loss


def make_meta_classification_val(loc, joint_train_kws=False):
    kws_params, kws_apply = load_kws_model(loc)
    vec_kws_apply = jax.vmap(kws_apply, in_axes=(None, None, 0))

    def meta_classification_val(
        losses, outputs, data_samples, metadata, outer_learnable
    ):
        if joint_train_kws:
            cur_kws_params = outer_learnable["kws_p"]
        else:
            cur_kws_params = kws_params

        out = jnp.reshape(
            outputs["out"],
            (outputs["out"].shape[0], -1, outputs["out"].shape[-1]),
        )

        target_class = metadata["label"]
        predicted_class = jnp.argmax(vec_kws_apply(cur_kws_params, None, out), axis=-1)

        acc = jnp.mean(predicted_class == target_class)
        return -acc

    return meta_classification_val


"""
For baseline Meta-AEC models with a pretrained KWS, set --outer_loss_alpha to 1.0. For CT-Meta-AEC models with a pretrained KWS, set --outer_loss_alpha to a value less than 1.

python ct_metaaec_kws.py --n_frames 4 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 16 --group_size 5 --group_hop 2 --h_size 48 --total_epochs 1000 --val_period 10 --reduce_lr_patience 1 --early_stop_patience 3 --name example_ct_metaaec --unroll 93 --kws_loc ./ckpts/35cmds_tcn.pkl --kws_mode 35cmds --lr 2e-4 --outer_loss_alpha 0.5 --debug

For joint trained models, add the --joint_train_kws flag. For models without AEC, add the --no_aec flag. For models without AEC training, add the --no_aec_train flag.

python ct_metaaec_kws.py --n_frames 4 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --n_devices 1 --batch_size 16 --group_size 5 --group_hop 2 --h_size 48 --total_epochs 1000 --val_period 5 --reduce_lr_patience 1 --early_stop_patience 5 --name example_jct_metaaec --unroll 93 --kws_loc ./ckpts/35cmds_tcn.pkl --kws_mode 35cmds --outer_loss_alpha 0.50 --joint_train_kws --use_kws_init_ckpt --use_aec_init_ckpt --aec_init_ckpt 5_kws_loss_35cmds_med_25_0 2023_03_27_13_42_45 300 --lr 1e-4 --max_norm 1 --b1 0.9 --debug
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--kws_loc", type=str, default="")
    parser.add_argument("--kws_mode", type=str, default="10cmds")
    parser.add_argument("--joint_train_kws", action="store_true")
    parser.add_argument("--no_aec", action="store_true")
    parser.add_argument("--no_aec_train", action="store_true")
    parser.add_argument("--outer_loss_alpha", type=float, default=0.25)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--real_data", action="store_true")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--b1", type=float, default=0.99)
    parser.add_argument("--max_norm", type=float, default=10)

    parser.add_argument("--use_aec_init_ckpt", action="store_true")
    parser.add_argument(
        "--aec_init_ckpt",
        nargs="+",
        default=["0_kws_loss_35cmds_med_25_0", "2023_04_17_14_48_24", "210"],
    )

    parser.add_argument("--use_kws_init_ckpt", action="store_true")
    parser.add_argument(
        "--kws_init_ckpt",
        nargs="+",
        default=["1_kws_loss_35cmds_no_aec_25_0", "2023_03_28_16_15_32", "140"],
    )

    parser = optimizer_hofgru_aug.HO_FGRU.add_args(parser)
    parser = AECOLS.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    opt_pkg = optimizer_hofgru_aug
    # make the callbacks
    callbacks = []
    if not kwargs["debug"]:
        callbacks = [
            CheckpointCallback(name=kwargs["name"], ckpt_base_dir=kwargs["ckpt_dir"]),
            WandBCallback(project="ct-metaaec", name=kwargs["name"], entity=None),
        ]

    kwargs["outsize"] = kwargs["n_in_chan"] * kwargs["n_frames"]
    outer_train_loss = make_meta_classification_loss(
        kwargs["kws_loc"],
        kwargs["outer_loss_alpha"],
        kwargs,
        joint_train_kws=kwargs["joint_train_kws"],
        joint_train_aec=not kwargs["no_aec_train"],
        no_aec=kwargs["no_aec"],
    )

    dset = RealKWSAECDataset if kwargs["real_data"] else KWSAECDataset

    train_loader = NumpyLoader(
        dset(max_len=48000, mode="train", kws_mode=kwargs["kws_mode"]),
        batch_size=kwargs["batch_size"],
        num_workers=4,
        persistent_workers=True,
        shuffle=True,
    )
    val_loader = NumpyLoader(
        dset(max_len=80000, mode="val", kws_mode=kwargs["kws_mode"]),
        num_workers=2,
        batch_size=kwargs["batch_size"],
    )
    test_loader = NumpyLoader(
        dset(max_len=80000, mode="test", kws_mode=kwargs["kws_mode"]),
        batch_size=kwargs["batch_size"],
    )

    system = MetaAFTrainer(
        _filter_fwd=_AECOLS_fwd,
        filter_kwargs=AECOLS.grab_args(kwargs),
        filter_loss=aec_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=opt_pkg._fwd,
        optimizer_kwargs=opt_pkg.HO_FGRU.grab_args(kwargs),
        meta_train_loss=outer_train_loss,
        meta_val_loss=make_meta_classification_val(
            kwargs["kws_loc"], joint_train_kws=kwargs["joint_train_kws"]
        ),
        init_optimizer=opt_pkg.init_optimizer_all_data,
        make_mapped_optimizer=opt_pkg.make_mapped_optimizer_all_data,
        make_train_mapped_optimizer=opt_pkg.make_train_mapped_optimizer_all_data,
        callbacks=callbacks,
        kwargs=kwargs,
    )

    outer_learnable = None
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    outer_learnable = system.init_outer_learnable(subkey)

    # load a pre-trained KWS module
    if kwargs["joint_train_kws"]:
        kws_p, _ = load_kws_model(kwargs["kws_loc"])
        outer_learnable["kws_p"] = kws_p

    # load a pre-trained AEC module
    if kwargs["use_aec_init_ckpt"]:
        e = kwargs["aec_init_ckpt"][2]
        ckpt_loc = os.path.join(
            "./ckpts",
            kwargs["aec_init_ckpt"][0],
            kwargs["aec_init_ckpt"][1],
            f"epoch_{e}.pkl",
        )
        with open(ckpt_loc, "rb") as f:
            aec_init_ckpt = pickle.load(f)

        for k in aec_init_ckpt:
            outer_learnable[k] = aec_init_ckpt[k]

    if kwargs["use_kws_init_ckpt"]:
        e = kwargs["kws_init_ckpt"][2]
        ckpt_loc = os.path.join(
            "./ckpts",
            kwargs["kws_init_ckpt"][0],
            kwargs["kws_init_ckpt"][1],
            f"epoch_{e}.pkl",
        )
        with open(ckpt_loc, "rb") as f:
            kws_init_ckpt = pickle.load(f)
        outer_learnable["kws_p"] = kws_init_ckpt["kws_p"]

    # start the training
    outer_learned, losses = system.train(
        **MetaAFTrainer.grab_args(kwargs),
        meta_opt_kwargs={"step_size": kwargs["lr"], "b1": kwargs["b1"]},
        meta_opt_preprocess=Partial(clip_grads, max_norm=kwargs["max_norm"], eps=1e-9),
        outer_learnable=outer_learnable,
        key=key,
    )
