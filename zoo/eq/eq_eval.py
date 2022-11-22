import argparse
import pickle
import os
import json
import pprint
import tqdm
import soundfile as sf
import numpy as np

import jax
import jax.numpy as jnp
import torch
import torchaudio

import metaaf
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_gru
from metaaf.optimizer_gru import EGRU
import metaaf.optimizer_lms as lms
import metaaf.optimizer_nlms as nlms
import metaaf.optimizer_rmsprop as rms
import metaaf.optimizer_rls as rls

import zoo.eq.eq as eq
from zoo import metrics


def get_all_metrics(target, model_out, model_in):
    l = min(len(target), len(model_out), len(model_in))
    target, model_out, model_in = target[:l], model_out[:l], model_in[:l]

    res = {}
    res["snr"] = metrics.snr(model_out, target)

    in_res = {}
    in_res["snr"] = metrics.snr(model_in, target)

    delta_res = {f"delta_{k}": res[k] - in_res[k] for k in in_res}
    in_res = {f"in_{k}": in_res[k] for k in in_res}

    res.update(delta_res)
    res.update(in_res)
    return res


def system_snr(est_system, i, test_dataset, kwargs):
    # apply constraint if we had one
    if kwargs["constraint"] == "antialias":
        est_system_td = (
            jnp.fft.irfft(est_system)
            .at[(kwargs["window_size"] + kwargs["pad_size"]) // 2 :]
            .set(0.0)
        )

        est_system = jnp.fft.rfft(est_system_td)

    est_system_td = jnp.fft.irfft(est_system)[
        : (kwargs["window_size"] + kwargs["pad_size"]) // 2
    ]
    est_system = jnp.fft.rfft(est_system_td)

    # get the true system
    delta_torch = torch.zeros((kwargs["window_size"] + kwargs["pad_size"]) // 2, 1)
    delta_torch[0] = 1

    true_system = torchaudio.sox_effects.apply_effects_tensor(
        delta_torch, 16000, test_dataset.data[i]["effects"], channels_first=False
    )[0]

    true_system = true_system.numpy()
    true_system = jnp.fft.rfft(jnp.array(true_system[:, 0]))

    # measure the mag system error in dB
    EPS = 1e-10
    true_system = 1 / np.abs(true_system + EPS)

    snr = 10 * np.log10(
        np.mean(np.abs(true_system) ** 2)
        / (np.mean(np.abs(np.abs(true_system) - np.abs(est_system)) ** 2) + EPS)
        + EPS
    )
    return snr


def segmental_system_snr(system, outer_learnable, d_input, i, test_dataset, kwargs):
    online_step, state = system.make_online_infer(outer_learnable)
    hop_size = kwargs["hop_size"]
    system_snrs = []
    for hop in range(0, len(d_input["u"][0]) - 2 * hop_size, hop_size):

        cur_data = {
            "signals": {
                "u": d_input["u"][0, hop : hop + hop_size],
                "d": d_input["d"][0, hop : hop + hop_size],
            },
            "metadata": {},
        }
        _, _, state = online_step(state, cur_data, jax.random.PRNGKey(0))

        filter = jax.tree_util.tree_flatten(state[1])[0][0][0, :, 0]
        system_snrs.append(system_snr(filter, i, test_dataset, kwargs))
    return np.array(system_snrs)


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
        eq.EQDAPSDataset(mode="train", n_signals=2048), batch_size=1
    )

    outer_train_loss = eq.meta_log_mse_loss
    # switch case to find the right optimizer functions
    if kwargs["optimizer"] == "gru":
        optimizer_kwargs = EGRU.grab_args(kwargs)
        _optimizer_fwd = optimizer_gru._fwd
        init_optimizer = optimizer_gru.init_optimizer_all_data
        make_mapped_optmizer = optimizer_gru.make_mapped_optmizer_all_data

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
        _filter_fwd=eq._NOOPEQOLS_fwd if noop else eq._EQOLS_fwd,
        filter_kwargs=eq.NOOPEQOLS.grab_args(kwargs)
        if noop
        else eq.EQOLS.grab_args(kwargs),
        filter_loss=eq.eq_loss,
        optimizer_kwargs=optimizer_kwargs,
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
        meta_train_loss=outer_train_loss,
        meta_val_loss=eq.neg_snr_val_loss,
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
    test_dataset = eq.EQDAPSDataset(mode="test", n_signals=2048, is_fir=True)
    all_metrics = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        u = data["signals"]["u"][None]
        d = data["signals"]["d"][None]

        d_input = {"u": u, "d": d}
        pred, aux = system.infer(
            {"signals": d_input, "metadata": {}}, fit_infer=fit_infer
        )
        pred = np.array(pred[0, :, 0])

        # get metrics computed on outputs as well as on system
        output_metrics = get_all_metrics(d[0, :, 0], pred, u[0, :, 0])
        est_system = jax.tree_util.tree_flatten(aux[-1][1])[0][0][0, 0, :, 0]
        output_metrics["system_snr"] = system_snr(est_system, i, test_dataset, kwargs)
        output_metrics["seg_system_snr"] = segmental_system_snr(
            system, outer_learnable, d_input, i, test_dataset, kwargs
        )

        all_metrics.append(output_metrics)

        # possibly save the outputs
        if eval_kwargs["save_outputs"]:
            sf.write(os.path.join(out_dir, f"{i}_out.wav"), pred, fs)
            sf.write(os.path.join(out_dir, f"{i}_u.wav"), u[0, :, 0], fs)
            sf.write(os.path.join(out_dir, f"{i}_d.wav"), d[0, :, 0], fs)

    # print metrics
    try:
        mean_metrics = metrics.get_mean_metrics(all_metrics)
        std_metrics = metrics.get_std_metrics(all_metrics)

        for k in mean_metrics.keys():
            a_mean, a_sd = mean_metrics[k], std_metrics[k]
            print(f"{k}:{a_mean:.3f}+-{a_sd:.3f}")
    except:
        print("Failed to Compute Some Metrics")

    # if saving outputs save all the metrics
    if eval_kwargs["save_metrics"]:
        metrics_pkl = os.path.join(out_dir, "metrics.pkl")
        with open(metrics_pkl, "wb") as f:
            pickle.dump(all_metrics, f)
