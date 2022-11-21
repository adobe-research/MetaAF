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
import metaaf.optimizer_gru as gru
import metaaf.optimizer_fgru as fgru
import metaaf.optimizer_lms as lms
import metaaf.optimizer_nlms as nlms
import metaaf.optimizer_rmsprop as rms
import metaaf.optimizer_rls as rls

import zoo.gsc.gsc as gsc
from zoo import metrics


def get_all_metrics(clean, enhanced, mix, fs=16000):
    l = min(len(clean), len(enhanced), len(mix))
    clean, enhanced, mix = clean[:l], enhanced[:l], mix[:l]

    res = {}
    res["stoi"] = metrics.stoi(enhanced, clean, fs=fs)
    res["sisdr"] = metrics.sisdr(enhanced, clean)
    res["snr"] = metrics.snr(mix, enhanced)
    sdr, sir, sar = metrics.bss_eval_metrics(enhanced, clean, mix)
    res["sdr"] = sdr
    res["sir"] = sir
    res["sar"] = sar

    in_res = {}
    in_res["stoi"] = metrics.stoi(mix, clean, fs=fs)
    in_res["sisdr"] = metrics.sisdr(mix, clean)
    in_res["snr"] = metrics.snr(mix, mix)
    # sdr, sir, sar = metrics.bss_eval_metrics(mix, clean, mix)
    in_res["sdr"] = 0  # sdr
    in_res["sir"] = 0  # sir
    in_res["sar"] = 0  # sar

    delta_res = {f"delta_{k}": res[k] - in_res[k] for k in in_res}
    in_res = {f"in_{k}": in_res[k] for k in in_res}

    res.update(delta_res)
    res.update(in_res)
    return res


def get_system_ckpt(ckpt_dir, e, verbose=True, noop=False):
    ckpt_loc = os.path.join(ckpt_dir, f"epoch_{e}.pkl")
    with open(ckpt_loc, "rb") as f:
        outer_learnable = pickle.load(f)

    kwargs_loc = os.path.join(ckpt_dir, "all_kwargs.json")
    with open(kwargs_loc, "rb") as f:
        kwargs = json.load(f)

    if verbose:
        pprint.pprint(kwargs)

    train_loader = NumpyLoader(
        gsc.Chime3Dataset(
            mode="train",
            n_mics=kwargs["n_in_chan"],
            signal_len=128000,
        ),
        batch_size=1,
    )

    outer_train_loss = gsc.meta_log_mse_loss
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
        _filter_fwd=gsc._NOOPGSCOLA_fwd if noop else gsc._GSCOLA_fwd,
        filter_kwargs=gsc.NOOPGSCOLA.grab_args(kwargs)
        if noop
        else gsc.GSCOLA.grab_args(kwargs),
        filter_loss=gsc.gsc_loss,
        optimizer_kwargs=optimizer_kwargs,
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
        meta_train_loss=outer_train_loss,
        meta_val_loss=gsc.make_neg_sisdr_val(kwargs["window_size"], kwargs["hop_size"]),
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
    parser.add_argument("--static_speech_interfere", action="store_true")

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
    test_dataset = gsc.Chime3Dataset(
        mode="test",
        n_mics=kwargs["n_in_chan"],
        static_speech_interfere=eval_kwargs["static_speech_interfere"],
    )
    all_metrics = []
    for i in tqdm.tqdm(range(0, len(test_dataset))):
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
        static_speech_interfere = eval_kwargs["static_speech_interfere"]
        metrics_pkl = os.path.join(
            out_dir, f"metrics_directional_{static_speech_interfere}.pkl"
        )
        with open(metrics_pkl, "wb") as f:
            pickle.dump(all_metrics, f)
