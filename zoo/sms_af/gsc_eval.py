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
from metaaf.filter import (
    make_inner_passthrough,
)
from metaaf import optimizer_hofgru_simple
from metaaf import preprocess_utils
from metaaf import postprocess_utils

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
    get_loss,
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
        Chime3Dataset(
            mode="train",
            n_mics=kwargs["n_in_chan"],
            signal_len=128000,
        ),
        batch_size=1,
    )

    system = MetaAFTrainer(
        _filter_fwd=_GSCOLA_fwd,
        filter_kwargs=GSCOLA.grab_args(kwargs),
        filter_loss=gsc_loss,
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
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
        kwargs=kwargs,
    )
    system.outer_learnable = outer_learnable
    return system, kwargs, outer_learnable


"""
# self supervised 1 iter
python gsc_eval.py --name gsc_self_16 --date 2023_08_24_23_45_22 --epoch 60 --save_metrics
python gsc_eval.py --name gsc_self --date 2023_08_23_18_04_41 --epoch 20 --save_metrics
python gsc_eval.py --name gsc_self_64 --date 2023_08_24_15_30_58 --epoch 10 --save_metrics

# full supervised 1 iter
python gsc_eval.py --name gsc_sisdr_val_sisdr_16 --date 2023_08_27_17_05_51 --epoch 330 --save_metrics
python gsc_eval.py --name gsc_sisdr_val_sisdr --date 2023_08_23_18_05_19 --epoch 370 --save_metrics
python gsc_eval.py --name gsc_sisdr_val_sisdr_64 --date 2023_08_24_15_31_35 --epoch 360 --save_metrics

# full supervised 1.5 iter
python gsc_eval.py --name gsc_sisdr_val_sisdr_posterior_16 --date 2023_08_27_04_34_36 --epoch 520 --save_metrics
python gsc_eval.py --name gsc_sisdr_val_sisdr_posterior --date 2023_08_23_18_05_23 --epoch 820 --save_metrics
python gsc_eval.py --name gsc_sisdr_val_sisdr_posterior_64 --date 2023_08_24_15_32_23 --epoch 500 --save_metrics

# full supervised 2.5 iter
python gsc_eval.py --name gsc_sisdr_val_sisdr_2iter_posterior_16 --date 2023_08_28_16_45_48 --epoch 250 --save_metrics
python gsc_eval.py --name gsc_sisdr_val_sisdr_2iter_posterior --date 2023_08_24_15_27_10 --epoch 690 --save_metrics
python gsc_eval.py --name gsc_sisdr_val_sisdr_2iter_posterior_64 --date 2023_08_27_17_19_50 --epoch 430 --save_metrics
"""

if __name__ == "__main__":
    # get checkpoint description from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts/gsc")
    parser.add_argument("--dataset", type=str, default="chime")

    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_metrics", action="store_true")

    eval_kwargs = vars(parser.parse_args())
    pprint.pprint(eval_kwargs)

    # build the checkpoint path
    ckpt_loc = os.path.join(
        eval_kwargs["ckpt_dir"], eval_kwargs["name"], eval_kwargs["date"]
    )
    epoch = int(eval_kwargs["epoch"])

    # load the checkpoint and kwargs file
    system, kwargs, outer_learnable = get_system_ckpt(ckpt_loc, epoch)
    fit_infer = system.make_fit_infer(outer_learnable=outer_learnable)
    fs = 16000

    # build the outputs path
    if eval_kwargs["save_outputs"] or eval_kwargs["save_metrics"]:
        epoch_num = eval_kwargs["epoch"]
        out_dir = os.path.join(
            eval_kwargs["out_dir"],
            eval_kwargs["name"],
            eval_kwargs["date"],
            f"epoch_{epoch_num}",
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
