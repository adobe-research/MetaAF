import argparse
import pprint
import tqdm
import numpy as np
import itertools
import jax

from metaaf.callbacks import CheckpointCallback
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer

from metaaf import optimizer_nlms as nlms
from metaaf import optimizer_rls as rls

from zoo.gsc.gsc import (
    Chime3Dataset,
    GSCOLA,
    _GSCOLA_fwd,
    gsc_loss,
)
from gsc import (
    get_val,
)

"""
python tune_gsc_baseline.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --cov_init identity --cov_update oracle --exp_avg .9 --steer_method rank1 --cov_update_regularizer 0.01  --total_epochs 0 --batch_size 16 --n_devices 1  --inner_iterations 1 --val_metric neg_sisdr --optimizer nlms --name gsc_nlms

python tune_gsc_baseline.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --cov_init identity --cov_update oracle --exp_avg .9 --steer_method rank1 --cov_update_regularizer 0.01  --total_epochs 0 --batch_size 16 --n_devices 1  --inner_iterations 1 --auto_posterior --val_metric neg_sisdr --optimizer nlms --name gsc_nlms_15

python tune_gsc_baseline.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --cov_init identity --cov_update oracle --exp_avg .9 --steer_method rank1 --cov_update_regularizer 0.01  --total_epochs 0 --batch_size 16 --n_devices 1  --inner_iterations 2 --auto_posterior --val_metric neg_sisdr --optimizer nlms --name gsc_nlms_25

python tune_gsc_baseline.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --cov_init identity --cov_update oracle --exp_avg .9 --steer_method rank1 --cov_update_regularizer 0.01  --total_epochs 0 --batch_size 16 --n_devices 1  --inner_iterations 1 --val_metric neg_sisdr --optimizer rls --name gsc_rls

python tune_gsc_baseline.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --cov_init identity --cov_update oracle --exp_avg .9 --steer_method rank1 --cov_update_regularizer 0.01  --total_epochs 0 --batch_size 16 --n_devices 1  --inner_iterations 1 --auto_posterior --val_metric neg_sisdr --optimizer rls --name gsc_rls_15

python tune_gsc_baseline.py --n_frames 1 --window_size 1024 --hop_size 512 --n_in_chan 6 --n_out_chan 1 --is_real --cov_init identity --cov_update oracle --exp_avg .9 --steer_method rank1 --cov_update_regularizer 0.01  --total_epochs 0 --batch_size 16 --n_devices 1  --inner_iterations 2 --auto_posterior --val_metric neg_sisdr --optimizer rls --name gsc_rls_25
"""
if __name__ == "__main__":
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="kf")
    parser.add_argument("--val_metric", type=str, default="neg_sisdr")
    parser.add_argument("--name", type=str, default="")

    args_so_far, _ = parser.parse_known_args()

    gsc = GSCOLA
    _gsc_fwd = _GSCOLA_fwd
    parser = gsc.add_args(parser)

    if args_so_far.optimizer == "nlms":
        opt = nlms
    elif args_so_far.optimizer == "rls":
        opt = rls
    parser = opt.add_args(parser)

    parser = MetaAFTrainer.add_args(parser)
    kwargs = vars(parser.parse_args())
    pprint.pprint(kwargs)

    val_dataset = Chime3Dataset(
        mode="val",
        n_mics=kwargs["n_in_chan"],
        signal_len=128000,
    )

    test_dataset = Chime3Dataset(
        mode="val",
        n_mics=kwargs["n_in_chan"],
        signal_len=128000,
    )

    val_loader = NumpyLoader(
        val_dataset,
        batch_size=kwargs["batch_size"],
        num_workers=3,
    )
    test_loader = NumpyLoader(
        test_dataset,
        batch_size=kwargs["batch_size"],
        num_workers=1,
    )

    system = MetaAFTrainer(
        _filter_fwd=_gsc_fwd,
        filter_kwargs=gsc.grab_args(kwargs),
        filter_loss=gsc_loss,
        train_loader=val_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        _optimizer_fwd=opt._fwd,
        optimizer_kwargs=opt.grab_args(kwargs),
        meta_val_loss=get_val(kwargs),
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
    ckpt_cb = CheckpointCallback(name=kwargs["name"], ckpt_base_dir="./ckpts/gsc")
    ckpt_cb.on_init(
        system.inner_fixed, system.outer_fixed, system.kwargs, outer_learned
    )
    ckpt_cb.on_train_epoch_end(
        best_val_scores,
        best_val_scores,
        outer_learned,
        0,
    )
