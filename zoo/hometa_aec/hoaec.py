import numpy as np
import argparse
import jax
from metaaf import optimizer_hogru
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf.optimizer_hogru import HOElementWiseGRU
from metaaf.callbacks import CheckpointCallback, WandBCallback, AudioLoggerCallback

from zoo.aec.aec import (
    MSFTAECDataset,
    AECOLS,
    _AECOLS_fwd,
    aec_loss,
    meta_log_mse_loss,
    neg_erle_val_loss,
)


if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--double_talk", action="store_true")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--random_roll", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--no_extra", action="store_true")
    parser.add_argument("--b1", type=float, default=0.99)  # adam parameter

    parser = AECOLS.add_args(parser)
    parser = HOElementWiseGRU.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())

    if kwargs["group_mode"] == "diag":
        assert kwargs["group_size"] == 1
    else:
        assert kwargs["group_size"] > 1

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

    # make the dataloders
    train_loader = NumpyLoader(
        MSFTAECDataset(
            mode="train",
            double_talk=kwargs["double_talk"],
            random_roll=kwargs["random_roll"],
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
        ),
        batch_size=kwargs["batch_size"],
        num_workers=2,
    )
    test_loader = NumpyLoader(
        MSFTAECDataset(
            mode="test",
            double_talk=kwargs["double_talk"],
            random_roll=kwargs["random_roll"],
        ),
        batch_size=kwargs["batch_size"],
        num_workers=0,
    )

    if kwargs["no_extra"]:
        init_optimizer = optimizer_hogru.init_optimizer
        make_mapped_optmizer = optimizer_hogru.make_mapped_optmizer
    else:
        init_optimizer = optimizer_hogru.init_optimizer_all_data
        make_mapped_optmizer = optimizer_hogru.make_mapped_optmizer_all_data

    # make the callbacks
    callbacks = [
        CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./ckpts"),
        AudioLoggerCallback(name=kwargs["name"], outputs_base_dir="./outputs"),
        WandBCallback(project="higher_order_aec", name=kwargs["name"], entity=None),
    ]

    system = MetaAFTrainer(
        _filter_fwd=_AECOLS_fwd,
        filter_kwargs=AECOLS.grab_args(kwargs),
        filter_loss=aec_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=optimizer_hogru._elementwise_hogru_fwd,
        optimizer_kwargs=HOElementWiseGRU.grab_args(kwargs),
        meta_train_loss=meta_log_mse_loss,
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
        meta_opt_kwargs={"step_size": kwargs["lr"], "b1": kwargs["b1"]},
        key=key
    )
