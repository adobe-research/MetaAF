import argparse
import pickle
import os
import json
import pprint
import tqdm
import soundfile as sf
import numpy as np

from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_nlms as nlms
from metaaf import optimizer_rls as rls

import zoo.metrics as metrics
from zoo.gsc.gsc_eval import get_all_metrics
from zoo.gsc.gsc import (
    Chime3Dataset,
    GSCOLA,
    _GSCOLA_fwd,
    gsc_loss,
)
from gsc import (
    get_val,
)


def get_system_ckpt(ckpt_dir, e, verbose=True):
    ckpt_loc = os.path.join(ckpt_dir, f"epoch_{e}.pkl")
    with open(ckpt_loc, "rb") as f:
        outer_learnable = pickle.load(f)

    kwargs_loc = os.path.join(ckpt_dir, "all_kwargs.json")
    with open(kwargs_loc, "rb") as f:
        kwargs = json.load(f)

    if verbose:
        pprint.pprint(kwargs)

    if kwargs["optimizer"] == "nlms":
        opt_pkg = nlms
    elif kwargs["optimizer"] == "rls":
        opt_pkg = rls

    train_loader = NumpyLoader(
        Chime3Dataset(
            mode="train",
            n_mics=kwargs["n_in_chan"],
            signal_len=128000,
        ),
        batch_size=1,
    )

    gsc = GSCOLA
    _gsc_fwd = _GSCOLA_fwd
    system = MetaAFTrainer(
        _filter_fwd=_gsc_fwd,
        filter_kwargs=gsc.grab_args(kwargs),
        filter_loss=gsc_loss,
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
        _optimizer_fwd=opt_pkg._fwd,
        optimizer_kwargs=opt_pkg.grab_args(kwargs),
        meta_val_loss=get_val(kwargs),
        init_optimizer=opt_pkg.init_optimizer,
        make_mapped_optimizer=opt_pkg.make_mapped_optimizer,
        inner_iterations=kwargs["inner_iterations"],
        auto_posterior=kwargs["auto_posterior"],
        kwargs=kwargs,
    )
    system.outer_learnable = outer_learnable
    return system, kwargs, outer_learnable


"""
python baseline_gsc_eval.py --name gsc_nlms_25 --date 2023_08_31_22_24_25 --epoch 0 --save_metrics

python baseline_gsc_eval.py --name gsc_nlms_15 --date 2023_08_31_22_45_29 --epoch 0 --save_metrics

python baseline_gsc_eval.py --name gsc_nlms --date 2023_08_31_23_06_32 --epoch 0 --save_metrics

python baseline_gsc_eval.py --name gsc_rls_25 --date 2023_09_01_07_59_21 --epoch 0 --save_metrics

python baseline_gsc_eval.py --name gsc_rls_15 --date 2023_09_01_01_54_07 --epoch 0 --save_metrics

python baseline_gsc_eval.py --name gsc_rls --date 2023_09_01_01_43_57 --epoch 0 --save_metrics
"""
if __name__ == "__main__":
    # get checkpoint description from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0)

    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_metrics", action="store_true")

    eval_kwargs = vars(parser.parse_args())
    pprint.pprint(eval_kwargs)

    # build the checkpoint path
    ckpt_loc = os.path.join("./ckpts/gsc", eval_kwargs["name"], eval_kwargs["date"])
    epoch = int(eval_kwargs["epoch"])

    # load the checkpoint and kwargs file
    system, kwargs, outer_learnable = get_system_ckpt(
        ckpt_loc,
        epoch,
    )
    fit_infer = system.make_fit_infer(outer_learnable=outer_learnable)
    fs = 16000

    # build the outputs path
    if eval_kwargs["save_outputs"] or eval_kwargs["save_metrics"]:
        epoch_num = eval_kwargs["epoch"]
        out_dir = os.path.join(
            eval_kwargs["out_dir"],
            eval_kwargs["name"],
            eval_kwargs["date"],
            f"epoch_{epoch}",
        )
        os.makedirs(out_dir, exist_ok=True)

    # evaluate the model
    test_dataset = Chime3Dataset(
        mode="test",
        n_mics=kwargs["n_in_chan"],
    )

    all_metrics = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        m, s = data["signals"]["m"], data["signals"]["s"]

        d_input = {"m": m[None], "s": s[None]}
        pred = np.array(
            system.infer({"signals": d_input, "metadata": {}}, fit_infer=fit_infer)[0]
        )
        pred = pred[0, kwargs["window_size"] - kwargs["hop_size"] :, 0]

        all_metrics.append(get_all_metrics(s[:, 0], pred, m[:, 0], fs=fs))

        # possibly save the outputs
        if eval_kwargs["save_outputs"]:
            sf.write(os.path.join(out_dir, f"{i}_out.wav"), pred, fs)
            sf.write(os.path.join(out_dir, f"{i}_m.wav"), m[:, 0], fs)
            sf.write(os.path.join(out_dir, f"{i}_s.wav"), s[:, 0], fs)

    # print metrics
    mean_metrics = metrics.get_mean_metrics(all_metrics)
    std_metrics = metrics.get_std_metrics(all_metrics)

    for k in mean_metrics.keys():
        a_mean, a_sd = mean_metrics[k], std_metrics[k]
        print(f"{k}:{a_mean:.3f}+-{a_sd:.3f}")

    # if saving outputs save all the metrics
    if eval_kwargs["save_metrics"]:
        metrics_pkl = os.path.join(out_dir, "metrics.pkl")
        with open(metrics_pkl, "wb") as f:
            pickle.dump(all_metrics, f)
