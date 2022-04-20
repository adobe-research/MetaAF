import argparse
import pickle
import os
import json
import pprint
import tqdm
import soundfile as sf
import numpy as np

from jax.tree_util import Partial

import metaaf
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_gru
from metaaf.optimizer_gru import ElementWiseGRU
import metaaf.optimizer_lms as lms
import metaaf.optimizer_nlms as nlms
import metaaf.optimizer_rmsprop as rms
import metaaf.optimizer_rls as rls
import zoo.aec.optimizer_kf as kf

import zoo.aec.aec as aec
from zoo import metrics


def get_all_metrics(clean, enhanced, echo, mix, fs=16000):
    l = min(len(clean), len(enhanced), len(mix))
    clean, enhanced, mix = clean[:l], enhanced[:l], mix[:l]

    res = {}
    res["stoi"] = metrics.stoi(enhanced, clean, fs=fs)
    res["erle"] = metrics.erle(enhanced, mix, echo)
    res["serle"] = metrics.erle(enhanced, mix, echo, segmental=True)

    in_res = {}
    in_res["stoi"] = metrics.stoi(mix, clean, fs=fs)
    in_res["erle"] = metrics.erle(mix, mix, echo)
    in_res["serle"] = metrics.erle(mix, mix, echo, segmental=True)

    delta_res = {f"delta_{k}": res[k] - in_res[k] for k in in_res}
    in_res = {f"in_{k}": in_res[k] for k in in_res}

    res.update(delta_res)
    res.update(in_res)
    return res


def get_system_ckpt(ckpt_dir, e, model_type="egru", system_len=None, verbose=True):
    ckpt_loc = os.path.join(ckpt_dir, f"epoch_{e}.pkl")
    with open(ckpt_loc, "rb") as f:
        outer_learnable = pickle.load(f)

    kwargs_loc = os.path.join(ckpt_dir, "all_kwargs.json")
    with open(kwargs_loc, "rb") as f:
        kwargs = json.load(f)

    if system_len is not None:
        kwargs["window_size"] = system_len * 2
        kwargs["hop_size"] = system_len

    if verbose:
        pprint.pprint(kwargs)

    train_loader = NumpyLoader(aec.MSFTAECDataset(mode="train"), batch_size=1)

    if "outer_loss" not in kwargs or kwargs["outer_loss"] == "indep_mse":
        outer_train_loss = metaaf.optimizer_utils.frame_indep_meta_mse
    elif kwargs["outer_loss"] == "log_indep_mse":
        outer_train_loss = metaaf.optimizer_utils.frame_indep_meta_logmse
    elif kwargs["outer_loss"] == "self_mse":
        outer_train_loss = aec.meta_mse_loss
    elif kwargs["outer_loss"] == "log_self_mse":
        outer_train_loss = aec.meta_log_mse_loss
    # switch case to find the right optimizer functions
    if model_type == "egru":
        optimizer_kwargs = ElementWiseGRU.grab_args(kwargs)
        _optimizer_fwd = optimizer_gru._elementwise_gru_fwd

        if kwargs["extra_signals"] == "none":
            init_optimizer = optimizer_gru.init_optimizer
            make_mapped_optmizer = optimizer_gru.make_mapped_optmizer
        elif kwargs["extra_signals"] == "ude":
            init_optimizer = optimizer_gru.init_optimizer_all_data
            make_mapped_optmizer = optimizer_gru.make_mapped_optmizer_all_data

    elif model_type == "lms":
        optimizer_kwargs = lms.grab_args(kwargs)
        _optimizer_fwd = lms._fwd
        init_optimizer = lms.init_optimizer
        make_mapped_optmizer = lms.make_mapped_optmizer

    elif model_type == "nlms":
        optimizer_kwargs = nlms.grab_args(kwargs)
        _optimizer_fwd = nlms._fwd
        init_optimizer = nlms.init_optimizer
        make_mapped_optmizer = nlms.make_mapped_optmizer

    elif model_type == "rms":
        optimizer_kwargs = rms.grab_args(kwargs)
        _optimizer_fwd = rms._fwd
        init_optimizer = rms.init_optimizer
        make_mapped_optmizer = rms.make_mapped_optmizer

    elif model_type == "rls":
        optimizer_kwargs = rls.grab_args(kwargs)
        _optimizer_fwd = rls._fwd
        init_optimizer = rls.init_optimizer
        make_mapped_optmizer = rls.make_mapped_optmizer

    elif model_type == "kf":
        optimizer_kwargs = kf.grab_args(kwargs)
        _optimizer_fwd = kf._fwd
        init_optimizer = kf.init_optimizer
        make_mapped_optmizer = kf.make_mapped_optmizer

    system = MetaAFTrainer(
        _filter_fwd=aec._AECOLS_fwd,
        filter_kwargs=aec.AECOLS.grab_args(kwargs),
        filter_loss=aec.aec_loss,
        optimizer_kwargs=optimizer_kwargs,
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
        meta_train_loss=outer_train_loss,
        meta_val_loss=aec.neg_erle_val_loss,
        _optimizer_fwd=_optimizer_fwd,
        init_optimizer=init_optimizer,
        make_mapped_optmizer=make_mapped_optmizer,
        kwargs=kwargs,
    )
    system.outer_learnable = outer_learnable
    return system, kwargs, outer_learnable


if __name__ == "__main__":

    # get checkpoint description from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--model_type", type=str, default="egru")

    parser.add_argument("--out_dir", type=str, default="./taslp_outputs")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_metrics", action="store_true")
    parser.add_argument("--true_rir_len", type=int, default=None)
    parser.add_argument("--system_len", type=int, default=None)

    eval_kwargs = vars(parser.parse_args())
    pprint.pprint(eval_kwargs)

    # build the checkpoint path
    ckpt_loc = os.path.join("./taslp_ckpts", eval_kwargs["name"], eval_kwargs["date"])
    epoch = int(eval_kwargs["epoch"])

    # load the checkpoint and kwargs file
    system, kwargs, outer_learnable = get_system_ckpt(
        ckpt_loc,
        epoch,
        model_type=eval_kwargs["model_type"],
        system_len=eval_kwargs["system_len"],
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
            f"epoch_{epoch_num}",
        )
        os.makedirs(out_dir, exist_ok=True)

    if kwargs["dataset"] == "linear":
        aec_dataset = aec.MSFTAECDataset_RIR
        aec_dataset = Partial(aec_dataset, rir_len=eval_kwargs["true_rir_len"])
    elif kwargs["dataset"] == "nonlinear":
        aec_dataset = aec.MSFTAECDataset

    # evaluate the model
    test_dataset = aec_dataset(
        mode="test",
        double_talk=kwargs["double_talk"],
        scene_change=kwargs["scene_change"],
        random_roll=kwargs["random_roll"],
    )
    all_metrics = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        u, d, e, s = (
            data["signals"]["u"],
            data["signals"]["d"],
            data["signals"]["e"],
            data["signals"]["s"],
        )

        d_input = {"u": u[None], "d": d[None], "s": s[None], "e": e[None]}
        pred = system.infer({"signals": d_input, "metadata": {}}, fit_infer=fit_infer)[
            0
        ]
        pred = np.array(pred[0, :, 0])

        all_metrics.append(get_all_metrics(s[:, 0], pred, e[:, 0], d[:, 0], fs=fs))

        # possibly save the outputs
        if eval_kwargs["save_outputs"]:
            sf.write(os.path.join(out_dir, f"{i}_out.wav"), pred, fs)
            sf.write(os.path.join(out_dir, f"{i}_u.wav"), u[:, 0], fs)
            sf.write(os.path.join(out_dir, f"{i}_d.wav"), d[:, 0], fs)
            sf.write(os.path.join(out_dir, f"{i}_e.wav"), e[:, 0], fs)
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
        if kwargs["scene_change"]:
            metrics_pkl = os.path.join(out_dir, "metrics_scene_change.pkl")
        elif (
            eval_kwargs["true_rir_len"] is not None
            and eval_kwargs["system_len"] is None
        ):
            rir_len = eval_kwargs["true_rir_len"]
            metrics_pkl = os.path.join(out_dir, f"metrics_{rir_len}.pkl")
        elif (
            eval_kwargs["true_rir_len"] is not None
            and eval_kwargs["system_len"] is not None
        ):
            rir_len = eval_kwargs["true_rir_len"]
            system_len = eval_kwargs["system_len"]
            metrics_pkl = os.path.join(out_dir, f"metrics_{rir_len}_{system_len}.pkl")
        with open(metrics_pkl, "wb") as f:
            pickle.dump(all_metrics, f)
