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
from metaaf import optimizer_fgru
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


def get_system_ckpt(ckpt_dir, e, system_len=None, noop=False, verbose=False):
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
    if kwargs["optimizer"] == "gru":
        optimizer_kwargs = ElementWiseGRU.grab_args(kwargs)
        _optimizer_fwd = optimizer_gru._elementwise_gru_fwd

        if kwargs["extra_signals"] == "none":
            init_optimizer = optimizer_gru.init_optimizer
            make_mapped_optmizer = optimizer_gru.make_mapped_optmizer
        elif kwargs["extra_signals"] == "udey":
            init_optimizer = optimizer_gru.init_optimizer_all_data
            make_mapped_optmizer = optimizer_gru.make_mapped_optmizer_all_data

    elif kwargs["optimizer"] == "fgru":
        optimizer_kwargs = optimizer_fgru.TimeChanCoupledGRU.grab_args(kwargs)
        _optimizer_fwd = optimizer_fgru._timechancoupled_gru_fwd
        init_optimizer = optimizer_fgru.init_optimizer_all_data
        make_mapped_optmizer = optimizer_fgru.make_mapped_optmizer_all_data

    elif kwargs["optimizer"] == "lms":
        optimizer_kwargs = lms.grab_args(kwargs)
        _optimizer_fwd = lms._fwd
        init_optimizer = lms.init_optimizer
        make_mapped_optmizer = lms.make_mapped_optmizer

    elif kwargs["optimizer"] == "nlms":
        optimizer_kwargs = nlms.grab_args(kwargs)
        _optimizer_fwd = nlms._fwd
        init_optimizer = nlms.init_optimizer
        make_mapped_optmizer = nlms.make_mapped_optmizer

    elif kwargs["optimizer"] == "rms":
        optimizer_kwargs = rms.grab_args(kwargs)
        _optimizer_fwd = rms._fwd
        init_optimizer = rms.init_optimizer
        make_mapped_optmizer = rms.make_mapped_optmizer

    elif kwargs["optimizer"] == "rls":
        optimizer_kwargs = rls.grab_args(kwargs)
        _optimizer_fwd = rls._fwd
        init_optimizer = rls.init_optimizer
        make_mapped_optmizer = rls.make_mapped_optmizer

    elif kwargs["optimizer"] == "kf":
        optimizer_kwargs = kf.grab_args(kwargs)
        _optimizer_fwd = kf._fwd
        init_optimizer = kf.init_optimizer
        make_mapped_optmizer = kf.make_mapped_optmizer

    system = MetaAFTrainer(
        _filter_fwd=aec._NOOPAECOLS_fwd if noop else aec._AECOLS_fwd,
        filter_kwargs=aec.NOOPAECOLS.grab_args(kwargs)
        if noop
        else aec.AECOLS.grab_args(kwargs),
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


def run_eval(
    system, fit_infer, test_dataset, out_dir, test_dataset_name, eval_kwargs, fs=16000
):
    all_metrics = []
    N = len(test_dataset)
    for i in tqdm.tqdm(range(N)):
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
            base_path = os.path.join(out_dir, test_dataset_name)
            os.makedirs(base_path, exist_ok=True)
            sf.write(os.path.join(base_path, f"{i}_out.wav"), pred, fs)
            sf.write(os.path.join(base_path, f"{i}_u.wav"), u[:, 0], fs)
            sf.write(os.path.join(base_path, f"{i}_d.wav"), d[:, 0], fs)
            sf.write(os.path.join(base_path, f"{i}_e.wav"), e[:, 0], fs)
            sf.write(os.path.join(base_path, f"{i}_s.wav"), s[:, 0], fs)

    return all_metrics


if __name__ == "__main__":

    # get checkpoint description from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="./meta_ckpts")

    # get evaluation conditions from user
    parser.add_argument("--universal", action="store_true")
    parser.add_argument("--system_len", type=int, default=None)

    # these will only get set if universal is false
    parser.add_argument("--true_rir_len", type=int, default=None)

    # decide what to save
    parser.add_argument("--out_dir", type=str, default="./meta_outputs")
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
    system, kwargs, outer_learnable = get_system_ckpt(
        ckpt_loc,
        epoch,
        system_len=eval_kwargs["system_len"],
    )
    fit_infer = system.make_fit_infer(outer_learnable=outer_learnable)

    # build the outputs path
    out_dir = os.path.join(
        eval_kwargs["out_dir"],
        eval_kwargs["name"],
        eval_kwargs["date"],
        f"epoch_{epoch}",
    )
    if eval_kwargs["save_outputs"] or eval_kwargs["save_metrics"]:
        os.makedirs(out_dir, exist_ok=True)

    # name the filter and rir lengths
    true_rir_len = (
        "DEFAULT"
        if eval_kwargs["true_rir_len"] is None
        else eval_kwargs["true_rir_len"]
    )

    system_len = (
        "DEFAULT" if eval_kwargs["system_len"] is None else eval_kwargs["system_len"]
    )

    if eval_kwargs["universal"]:
        test_datasets = [
            aec.MSFTAECDataset_RIR(
                mode="test",
                double_talk=False,
                scene_change=False,
                random_roll=kwargs["random_roll"],
                random_level=kwargs["random_level"],
            ),
            aec.MSFTAECDataset_RIR(
                mode="test",
                double_talk=True,
                scene_change=False,
                random_roll=kwargs["random_roll"],
                random_level=kwargs["random_level"],
            ),
            aec.MSFTAECDataset_RIR(
                mode="test",
                double_talk=True,
                scene_change=True,
                random_roll=kwargs["random_roll"],
                random_level=kwargs["random_level"],
            ),
            aec.MSFTAECDataset(
                mode="test",
                double_talk=True,
                scene_change=False,
                random_roll=kwargs["random_roll"],
                random_level=kwargs["random_level"],
            ),
            aec.MSFTAECDataset(
                mode="test",
                double_talk=True,
                scene_change=True,
                random_roll=kwargs["random_roll"],
                random_level=kwargs["random_level"],
            ),
        ]

        test_dataset_names = [
            f"DT_{False}_SC_{False}_NL_{False}_TR_{true_rir_len}_SL_{system_len}",
            f"DT_{True}_SC_{False}_NL_{False}_TR_{true_rir_len}_SL_{system_len}",
            f"DT_{True}_SC_{True}_NL_{False}_TR_{true_rir_len}_SL_{system_len}",
            f"DT_{True}_SC_{False}_NL_{True}_TR_{true_rir_len}_SL_{system_len}",
            f"DT_{True}_SC_{True}_NL_{True}_TR_{true_rir_len}_SL_{system_len}",
        ]
    else:
        test_datasets = [
            aec.MSFTAECDataset_RIR(
                mode="test",
                double_talk=False,
                scene_change=False,
                random_roll=kwargs["random_roll"],
                rir_len=eval_kwargs["true_rir_len"],
            )
        ]

        test_dataset_names = [
            f"DT_{False}_SC_{False}_NL_{False}_TR_{true_rir_len}_SL_{system_len}",
        ]

    for i, test_dataset in enumerate(test_datasets):
        print(" --- Testing Dataset ---")
        print(f" --- {test_dataset_names[i]} ---")

        all_metrics = run_eval(
            system, fit_infer, test_dataset, out_dir, test_dataset_names[i], eval_kwargs
        )

        # print metrics
        mean_metrics = metrics.get_mean_metrics(all_metrics)
        std_metrics = metrics.get_std_metrics(all_metrics)

        for k in mean_metrics.keys():
            a_mean, a_sd = mean_metrics[k], std_metrics[k]
            print(f"{k}:{a_mean:.3f}+-{a_sd:.3f}")

        # if saving outputs save all the metrics
        if eval_kwargs["save_metrics"]:
            base_path = os.path.join(out_dir, test_dataset_names[i])
            os.makedirs(base_path, exist_ok=True)
            metrics_dir = os.path.join(base_path, "metrics.pkl")
            with open(metrics_dir, "wb") as f:
                pickle.dump(all_metrics, f)
