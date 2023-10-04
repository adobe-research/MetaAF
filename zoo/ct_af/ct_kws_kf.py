import argparse
import numpy as np
import itertools
import tqdm

import jax

from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf.callbacks import CheckpointCallback

import zoo.aec.aec as aec
from train_kws import KWSAECDataset
from ct_metaaec_kws import make_meta_classification_val
import optimizer_kf as kf


"""
python ct_kws_kf.py --n_frames 4 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --batch_size 32 --total_epochs 0 --n_devices 1 --name kf_35_25_0 --kws_loc ./ckpts/35cmds_tcn.pkl --kws_mode 35cmds
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="kf")
    parser.add_argument("--kws_loc", type=str, default="")
    parser.add_argument("--kws_mode", type=str, default="35cmds")

    parser = kf.add_args(parser)

    # get everything else
    parser.add_argument("--name", type=str, default="")

    parser = aec.AECOLS.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

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
        _filter_fwd=aec._AECOLS_fwd,
        filter_kwargs=aec.AECOLS.grab_args(kwargs),
        filter_loss=aec.aec_loss,
        train_loader=val_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=kf._fwd,
        optimizer_kwargs=kf.grab_args(kwargs),
        meta_val_loss=make_meta_classification_val(
            kwargs["kws_loc"], joint_train_kws=False
        ),
        init_optimizer=kf.init_optimizer,
        make_mapped_optimizer=kf.make_mapped_optimizer,
        kwargs=kwargs,
    )

    # epochs set to zero, this is just an initialization
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(**MetaAFTrainer.grab_args(kwargs), key=key)

    # now run the actual val loop
    tuning_options = kf.get_tuning_options(**kf.grab_args(kwargs))
    pprint.pprint(tuning_options)
    all_configs = list(itertools.product(*list(tuning_options.values())))

    # holder for best config
    best_config = dict(zip(list(tuning_options.keys()), all_configs[0]))
    best_val_scores = np.array([np.inf])

    # final holder
    mean_val_scores = []

    for config in tqdm.tqdm(all_configs):
        config_dict = dict(zip(list(tuning_options.keys()), config))
        outer_learned["optimizer_p"].update(config_dict)
        val_loss = system.val_loop(outer_learnable=outer_learned, early_exit_index=32)

        mean_val_scores.append(np.nanmean(val_loss))

        if np.nanmean(val_loss) < np.nanmean(best_val_scores):
            best_config = config_dict
            best_val_scores = val_loss

    for i in range(len(all_configs)):
        print(f"CFG: {all_configs[i]} -- {mean_val_scores[i]}")

    # run test and save the best parameters
    outer_learned["optimizer_p"].update(best_config)
    print(f"BEST -- CFG: {best_config} -- {np.median(best_val_scores)} --")

    test_loss = system.test_loop(outer_learnable=outer_learned)

    # print results without nans and percent nan
    percent_nan = np.isnan(test_loss).mean()
    print(
        f"Mean Test:{np.nanmean(test_loss)} - Median Test:{np.nanmedian(test_loss)} - % NAN {percent_nan}"
    )

    # save this model in the metaaf format
    ckpt_cb = CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./ckpts")
    ckpt_cb.on_init(
        system.inner_fixed, system.outer_fixed, system.kwargs, outer_learned
    )
    ckpt_cb.on_train_epoch_end(
        best_val_scores,
        best_val_scores,
        outer_learned,
        0,
    )
