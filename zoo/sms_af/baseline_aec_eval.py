import argparse
import json
import os
import pickle
import pprint

import numpy as np
import soundfile as sf
import tqdm

from metaaf import optimizer_nlms as nlms
import zoo.metrics as metrics
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf.filter import make_inner_grad
from metaaf import postprocess_utils
from zoo.aec.aec import MSFTAECDataset, aec_loss

import optimizer_kf as kf
from datasets import MSFTAECRealDataset
from aec_eval import get_all_metrics
from aec import (
    get_val_loss,
    AECOLA,
    _AECOLA_fwd,
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

    train_loader = NumpyLoader(
        MSFTAECDataset(mode="train", double_talk=True, random_roll=True), batch_size=1
    )

    _filt_fwd = _AECOLA_fwd
    filter_grab_args = AECOLA.grab_args

    if kwargs["optimizer"] == "kf":
        opt_pkg = kf
    elif kwargs["optimizer"] == "nlms":
        opt_pkg = nlms

    system = MetaAFTrainer(
        _filter_fwd=_filt_fwd,
        filter_kwargs=filter_grab_args(kwargs),
        filter_loss=aec_loss,
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
        _optimizer_fwd=opt_pkg._fwd,
        optimizer_kwargs=opt_pkg.grab_args(kwargs),
        meta_val_loss=get_val_loss(kwargs),
        init_optimizer=opt_pkg.init_optimizer,
        make_mapped_optimizer=opt_pkg.make_mapped_optimizer,
        make_get_filter_features=make_inner_grad,
        inner_iterations=kwargs["inner_iterations"],
        auto_posterior=kwargs["auto_posterior"],
        _postprocess_fwd=postprocess_utils._identity_fwd,
        postprocess_kwargs={},
        callbacks=[],
        kwargs=kwargs,
    )
    system.outer_learnable = outer_learnable
    return system, kwargs, outer_learnable


"""
python baseline_aec_eval.py --name kf_25 --date 2023_08_22_07_49_04 --epoch 0 --save_metrics

python baseline_aec_eval.py --name kf_15 --date 2023_08_21_22_33_32 --epoch 0 --save_metrics

python baseline_aec_eval.py --name kf --date 2023_08_21_22_28_11 --epoch 0 --save_metrics

python baseline_aec_eval.py --name nlms_25 --date 2023_08_22_06_17_10 --epoch 0 --save_metrics

python baseline_aec_eval.py --name nlms_15 --date 2023_08_21_21_54_34 --epoch 0 --save_metrics

python baseline_aec_eval.py --name nlms --date 2023_08_21_21_51_24 --epoch 0 --save_metrics
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
    parser.add_argument("--real_dataset", action="store_true")

    eval_kwargs = vars(parser.parse_args())
    pprint.pprint(eval_kwargs)

    # build the checkpoint path
    ckpt_loc = os.path.join("./ckpts/aec", eval_kwargs["name"], eval_kwargs["date"])
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
    if eval_kwargs["real_dataset"]:
        test_dataset = MSFTAECRealDataset(
            mode="test_real",
        )
    else:
        test_dataset = MSFTAECDataset(
            mode="test",
            double_talk=True,
            random_roll=False,
            random_level=False,
            scene_change=False,
        )
    test_batch_size = 16
    test_loader = NumpyLoader(
        test_dataset,
        batch_size=test_batch_size,
        drop_last=False,
        shuffle=False,
        num_workers=2,
    )

    all_metrics = []
    for i, data in enumerate(tqdm.tqdm(test_loader)):
        pred = system.infer(data, fit_infer=fit_infer)[0]
        pred = pred[..., 0]

        for j in range(pred.shape[0]):
            cur_pred = np.array(pred[j])
            cur_pred = pred[j, kwargs["window_size"] - kwargs["hop_size"] :]

            signals = data["signals"]
            all_metrics.append(
                get_all_metrics(
                    signals["s"][j, :, 0],
                    cur_pred,
                    signals["e"][j, :, 0],
                    signals["d"][j, :, 0],
                    signals["u"][j, :, 0],
                    fs=fs,
                )
            )

            # possibly save the outputs
            if eval_kwargs["save_outputs"]:
                sf.write(os.path.join(out_dir, f"{i}_out.wav"), cur_pred, fs)
                sf.write(os.path.join(out_dir, f"{i}_u.wav"), signals["u"][j, :, 0], fs)
                sf.write(os.path.join(out_dir, f"{i}_d.wav"), signals["d"][j, :, 0], fs)
                sf.write(os.path.join(out_dir, f"{i}_e.wav"), signals["e"][j, :, 0], fs)
                sf.write(os.path.join(out_dir, f"{i}_s.wav"), signals["s"][j, :, 0], fs)

    # print metrics
    mean_metrics = metrics.get_mean_metrics(all_metrics)
    std_metrics = metrics.get_std_metrics(all_metrics)

    for k in mean_metrics.keys():
        a_mean, a_sd = mean_metrics[k], std_metrics[k]
        print(f"{k}:{a_mean:.3f}+-{a_sd:.3f}")

    # if saving outputs save all the metrics
    if eval_kwargs["save_metrics"]:
        metrics_pkl = os.path.join(out_dir, f"metrics.pkl")

        if eval_kwargs["real_dataset"]:
            metrics_pkl = os.path.join(out_dir, f"metrics_real.pkl")

        with open(metrics_pkl, "wb") as f:
            pickle.dump(all_metrics, f)
