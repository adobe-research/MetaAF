import argparse
import numpy as np
import itertools
import tqdm

import jax

from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
import metaaf.optimizer_lms as lms
import metaaf.optimizer_nlms as nlms
import metaaf.optimizer_rmsprop as rms
import metaaf.optimizer_rls as rls
from metaaf.callbacks import CheckpointCallback

import zoo.eq.eq as eq

"""
Sample command to tune a baseline:
python eq_baselines.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 1 --n_out_chan 1 --is_real --batch_size 32 --total_epochs 0 --n_devices 1 --name eq_antialias_nlms --optimizer nlms --constraint antialias
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="lms")

    # get the optimizer package
    optimizer_type = parser.parse_known_args()[0].optimizer
    if optimizer_type == "lms":
        optimizer_pkg = lms
    elif optimizer_type == "nlms":
        optimizer_pkg = nlms
    elif optimizer_type == "rms":
        optimizer_pkg = rms
    elif optimizer_type == "rls":
        optimizer_pkg = rls

    # load its arguments
    parser = optimizer_pkg.add_args(parser)

    # get everything else
    parser.add_argument("--name", type=str, default="")
    parser = eq.EQOLS.add_args(parser)
    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    val_loader = NumpyLoader(
        eq.EQDAPSDataset(mode="val", n_signals=2048),
        batch_size=kwargs["batch_size"],
        num_workers=2,
    )
    test_loader = NumpyLoader(
        eq.EQDAPSDataset(mode="test", n_signals=2048, is_fir=True),
        batch_size=kwargs["batch_size"],
        num_workers=0,
    )

    system = MetaAFTrainer(
        _filter_fwd=eq._EQOLS_fwd,
        filter_kwargs=eq.EQOLS.grab_args(kwargs),
        filter_loss=eq.eq_loss,
        train_loader=val_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=optimizer_pkg._fwd,
        optimizer_kwargs=optimizer_pkg.grab_args(kwargs),
        meta_val_loss=eq.neg_snr_val_loss,
        init_optimizer=optimizer_pkg.init_optimizer,
        make_mapped_optmizer=optimizer_pkg.make_mapped_optmizer,
        kwargs=kwargs,
    )

    # epochs set to zero, this is just an initialization
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(**MetaAFTrainer.grab_args(kwargs), key=key)

    # now run the actual val loop
    tuning_options = optimizer_pkg.get_tuning_options(**optimizer_pkg.grab_args(kwargs))
    pprint.pprint(tuning_options)
    all_configs = list(itertools.product(*list(tuning_options.values())))

    # holder for best config
    best_config = dict(zip(list(tuning_options.keys()), all_configs[0]))
    best_val_scores = np.array([np.inf])

    # final holder
    median_val_scores = []

    for config in tqdm.tqdm(all_configs):
        config_dict = dict(zip(list(tuning_options.keys()), config))
        outer_learned["optimizer_p"].update(config_dict)
        val_loss = system.val_loop(outer_learnable=outer_learned)

        median_val_scores.append(np.median(val_loss))

        if np.median(val_loss) < np.median(best_val_scores):
            best_config = config_dict
            best_val_scores = val_loss

    for i in range(len(all_configs)):
        print(f"CFG: {all_configs[i]} -- {median_val_scores[i]}")

    # run test and save the best parameters
    outer_learned["optimizer_p"].update(best_config)
    print(f"BEST -- CFG: {best_config} -- {np.median(best_val_scores)} --")

    test_loss = system.test_loop(outer_learnable=outer_learned)

    # print results without nans and percent nan
    percent_nan = np.isnan(test_loss).mean()
    print(
        f"Mean Test:{np.nanmean(test_loss)} - Median Test:{np.nanmedian(test_loss)} - % NAN {percent_nan}"
    )

    # save this model in the autodsp format
    ckpt_cb = CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./taslp_ckpts")
    ckpt_cb.on_init(
        system.inner_fixed, system.outer_fixed, system.kwargs, outer_learned
    )
    ckpt_cb.on_train_epoch_end(
        best_val_scores,
        best_val_scores,
        outer_learned,
        0,
    )
