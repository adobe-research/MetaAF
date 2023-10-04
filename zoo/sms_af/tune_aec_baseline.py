import argparse
import numpy as np
import itertools
import tqdm

import jax

from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf.callbacks import CheckpointCallback
from metaaf import optimizer_nlms as nlms
from zoo.aec.aec import (
    aec_loss,
    MSFTAECDataset,
)

from aec import get_val_loss
import optimizer_kf as kf
from aec import AECOLA, _AECOLA_fwd

"""
python tune_aec_baseline.py --n_frames 8 --window_size 512 --hop_size 256 --n_in_chan 1 --n_out_chan 1 --is_real --no_analysis_window --batch_size 32 --total_epochs 0 --n_devices 1  --inner_iterations 2 --auto_posterior --val_loss neg_sisdr_ola --optimizer kf --name kf_25

python tune_aec_baseline.py --n_frames 8 --window_size 512 --hop_size 256 --n_in_chan 1 --n_out_chan 1 --is_real --no_analysis_window --batch_size 32 --total_epochs 0 --n_devices 1  --inner_iterations 1 --auto_posterior --val_loss neg_sisdr_ola --optimizer kf --name kf_15

python tune_aec_baseline.py --n_frames 8 --window_size 512 --hop_size 256 --n_in_chan 1 --n_out_chan 1 --is_real --no_analysis_window --batch_size 32 --total_epochs 0 --n_devices 1  --inner_iterations 1 --val_loss neg_sisdr_ola --optimizer kf --name kf

python tune_aec_baseline.py --n_frames 8 --window_size 512 --hop_size 256 --n_in_chan 1 --n_out_chan 1 --is_real --no_analysis_window --batch_size 32 --total_epochs 0 --n_devices 1  --inner_iterations 2 --auto_posterior --val_loss neg_sisdr_ola --optimizer nlms --name nlms_25

python tune_aec_baseline.py --n_frames 8 --window_size 512 --hop_size 256 --n_in_chan 1 --n_out_chan 1 --is_real --no_analysis_window --batch_size 32 --total_epochs 0 --n_devices 1  --inner_iterations 1 --auto_posterior --val_loss neg_sisdr_ola --optimizer nlms --name nlms_15

python tune_aec_baseline.py --n_frames 8 --window_size 512 --hop_size 256 --n_in_chan 1 --n_out_chan 1 --is_real --no_analysis_window --batch_size 32 --total_epochs 0 --n_devices 1  --inner_iterations 1 --val_loss neg_sisdr_ola --optimizer nlms --name nlms
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="kf")
    parser.add_argument("--val_loss", type=str, default="neg_sisdr")
    parser.add_argument("--name", type=str, default="")

    args_so_far, _ = parser.parse_known_args()

    aec = AECOLA
    _aec_fwd = _AECOLA_fwd
    parser = aec.add_args(parser)

    if args_so_far.optimizer == "kf":
        opt = kf
    elif args_so_far.optimizer == "nlms":
        opt = nlms
    parser = opt.add_args(parser)

    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    val_dataset = MSFTAECDataset(
        mode="val",
        double_talk=True,
        random_roll=True,
        random_level=True,
        max_len=160000,
    )

    test_dataset = MSFTAECDataset(
        mode="test",
        double_talk=True,
        random_roll=False,
        random_level=False,
        scene_change=False,
    )

    val_loader = NumpyLoader(
        val_dataset,
        batch_size=kwargs["batch_size"],
        num_workers=2,
    )
    test_loader = NumpyLoader(
        test_dataset,
        batch_size=kwargs["batch_size"],
        num_workers=2,
    )

    system = MetaAFTrainer(
        _filter_fwd=_aec_fwd,
        filter_kwargs=aec.grab_args(kwargs),
        filter_loss=aec_loss,
        train_loader=val_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=opt._fwd,
        optimizer_kwargs=opt.grab_args(kwargs),
        meta_val_loss=get_val_loss(kwargs),
        init_optimizer=opt.init_optimizer,
        make_mapped_optimizer=opt.make_mapped_optimizer,
        kwargs=kwargs,
    )

    # epochs set to zero, this is just an initialization
    key = jax.random.PRNGKey(0)
    outer_learned, losses = system.train(**MetaAFTrainer.grab_args(kwargs), key=key)

    # now run the actual val loop
    tuning_options = opt.get_tuning_options(**opt.grab_args(kwargs))
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
        val_loss = system.val_loop(outer_learnable=outer_learned, early_exit_index=5)

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
    ckpt_cb = CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./ckpts/aec")
    ckpt_cb.on_init(
        system.inner_fixed, system.outer_fixed, system.kwargs, outer_learned
    )
    ckpt_cb.on_train_epoch_end(
        best_val_scores,
        best_val_scores,
        outer_learned,
        0,
    )
