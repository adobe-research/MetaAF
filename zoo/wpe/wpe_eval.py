import argparse
import pickle
import os
import json
import pprint
import tqdm
import soundfile as sf
import numpy as np

import metaaf
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
import metaaf.optimizer_gru as gru
import metaaf.optimizer_fgru as fgru
import metaaf.optimizer_lms as lms
import metaaf.optimizer_nlms as nlms
import metaaf.optimizer_rmsprop as rms
import metaaf.optimizer_rls as rls

import zoo.wpe.wpe as wpe
from zoo import metrics


def get_all_metrics(clean, enhanced, mix, fs=16000):
    l = min(len(clean), len(enhanced), len(mix))
    clean, enhanced, mix = clean[:l], enhanced[:l], mix[:l]

    res = {}
    res["stoi"] = metrics.stoi(enhanced, clean, fs=fs)
    res["fwssnr"] = metrics.fwssnr(enhanced, clean, fs=fs)
    res["srr"] = metrics.srr_stft(enhanced, mix)
    res["seg_srr"] = metrics.srr_stft(enhanced, mix, segmental=True)

    in_res = {}
    in_res["stoi"] = metrics.stoi(mix, clean, fs=fs)
    in_res["fwssnr"] = metrics.fwssnr(mix, clean, fs=fs)
    in_res["srr"] = metrics.srr_stft(mix, mix)

    delta_res = {f"delta_{k}": res[k] - in_res[k] for k in in_res}
    in_res = {f"in_{k}": in_res[k] for k in in_res}

    res.update(delta_res)
    res.update(in_res)
    return res


def get_system_ckpt(ckpt_dir, e, noop=False, verbose=True):
    ckpt_loc = os.path.join(ckpt_dir, f"epoch_{e}.pkl")
    with open(ckpt_loc, "rb") as f:
        outer_learnable = pickle.load(f)

    kwargs_loc = os.path.join(ckpt_dir, "all_kwargs.json")
    with open(kwargs_loc, "rb") as f:
        kwargs = json.load(f)

    if verbose:
        pprint.pprint(kwargs)

    train_loader = NumpyLoader(
        wpe.ReverbDataset(
            mode="train",
            n_mics=kwargs["n_in_chan"],
            signal_len=128000,
        ),
        batch_size=1,
    )

    outer_train_loss = wpe.make_meta_log_mse_loss(
        kwargs["window_size"], kwargs["hop_size"]
    )

    # switch case to find the right optimizer functions
    if kwargs["optimizer"] == "egru":
        optimizer_kwargs = gru.EGRU.grab_args(kwargs)
        _optimizer_fwd = gru._fwd
        init_optimizer = gru.init_optimizer_all_data
        make_mapped_optmizer = gru.make_mapped_optmizer_all_data

    elif kwargs["optimizer"] == "fgru":
        optimizer_kwargs = fgru.FGRU.grab_args(kwargs)
        _optimizer_fwd = fgru._fwd
        init_optimizer = fgru.init_optimizer_all_data
        make_mapped_optmizer = fgru.make_mapped_optmizer_all_data

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

    system = MetaAFTrainer(
        _filter_fwd=wpe._NOOPWPEOLA_fwd if noop else wpe._WPEOLA_fwd,
        filter_kwargs=wpe.NOOPWPEOLA.grab_args(kwargs)
        if noop
        else wpe.WPEOLA.grab_args(kwargs),
        filter_loss=wpe.dereverb_loss,
        optimizer_kwargs=optimizer_kwargs,
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
        meta_train_loss=outer_train_loss,
        meta_val_loss=wpe.make_srr_val(kwargs["window_size"], kwargs["hop_size"]),
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
    parser.add_argument("--ckpt_dir", type=str, default="./meta_ckpts")

    parser.add_argument("--out_dir", type=str, default="./meta_outputs")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_metrics", action="store_true")
    parser.add_argument("--scene_change", action="store_true")

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
    test_dataset = wpe.ReverbDataset(mode="test", n_mics=kwargs["n_in_chan"])
    all_metrics = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        u, d = data["signals"]["u"], data["signals"]["d"]

        d_input = {"u": u[None], "d": d[None]}
        pred = np.array(
            system.infer({"signals": d_input, "metadata": {}}, fit_infer=fit_infer)[0]
        )
        pred = pred[0, kwargs["window_size"] - kwargs["hop_size"] :, 0]

        all_metrics.append(get_all_metrics(u[:, 0], pred, d[:, 0], fs=fs))

        # possibly save the outputs
        if eval_kwargs["save_outputs"]:
            sf.write(os.path.join(out_dir, f"{i}_out.wav"), pred, fs)
            sf.write(os.path.join(out_dir, f"{i}_u.wav"), u[:, 0], fs)
            sf.write(os.path.join(out_dir, f"{i}_d.wav"), d[:, 0], fs)

    # print metrics
    for k in all_metrics[0].keys():
        try:
            a_mean = np.nanmean([m[k] for m in all_metrics])
            a_sd = np.nanstd([m[k] for m in all_metrics])
            print(f"{k}:{a_mean:.3f}+-{a_sd:.3f}")
        except:
            print(f"Failed to Compute {k} Means")

    # if saving outputs save all the metrics
    if eval_kwargs["save_metrics"]:
        metrics_pkl = os.path.join(out_dir, "metrics.pkl")
        with open(metrics_pkl, "wb") as f:
            pickle.dump(all_metrics, f)
