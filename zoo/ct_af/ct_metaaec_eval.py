import argparse
import pickle
import os
import json
import pprint
import tqdm
import soundfile as sf
import numpy as np
import jax.numpy as jnp
import datetime
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_hofgru_aug

from zoo.aec.aec import AECOLS, _AECOLS_fwd, aec_loss
from zoo import metrics

from train_kws import (
    RealKWSAECDataset,
    KWSAECDataset,
    load_kws_model,
)
from ct_metaaec_kws import (
    make_meta_classification_loss,
    make_meta_classification_val,
)


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


def get_system_ckpt(ckpt_dir, e, kws_loc, kws_mode, verbose=False):
    ckpt_loc = os.path.join(ckpt_dir, f"epoch_{e}.pkl")
    with open(ckpt_loc, "rb") as f:
        outer_learnable = pickle.load(f)

    kwargs_loc = os.path.join(ckpt_dir, "all_kwargs.json")
    with open(kwargs_loc, "rb") as f:
        kwargs = json.load(f)

    if kws_loc is not None and kws_mode is not None:
        kwargs["kws_loc"] = kws_loc
        kwargs["kws_mode"] = kws_mode

    if verbose:
        pprint.pprint(kwargs)

    train_loader = NumpyLoader(
        KWSAECDataset(max_len=64000, mode="train", kws_mode=kwargs["kws_mode"]),
        batch_size=1,
    )

    _aec_fwd = _AECOLS_fwd
    aec_grab_args = AECOLS.grab_args
    opt_pkg = optimizer_hofgru_aug

    system = MetaAFTrainer(
        _filter_fwd=_aec_fwd,
        filter_kwargs=aec_grab_args(kwargs),
        filter_loss=aec_loss,
        optimizer_kwargs=opt_pkg.HO_FGRU.grab_args(kwargs),
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
        meta_train_loss=make_meta_classification_loss(
            kwargs["kws_loc"], kwargs["outer_loss_alpha"], kwargs
        ),
        meta_val_loss=make_meta_classification_val(kwargs["kws_loc"]),
        _optimizer_fwd=opt_pkg._fwd,
        init_optimizer=opt_pkg.init_optimizer_all_data,
        make_mapped_optimizer=opt_pkg.make_mapped_optimizer_all_data,
        inner_iterations=1,
        auto_posterior=False,
        kwargs=kwargs,
    )
    system.outer_learnable = outer_learnable
    return system, kwargs, outer_learnable


"""
To run evaluation, plug in the desired checkpoints information and set the kws mode/model. If you used joint training, the joint trained KWS checkpoint will automatically be selected.

python ct_metaaec_eval.py --name 5_kws_loss_35cmds_med_25_0 --date 2023_03_27_13_42_45 --epoch 300 --kws_loc ./ckpts/35cmds_tcn.pkl --kws_mode 35cmds
"""
if __name__ == "__main__":
    # get checkpoint description from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")

    # configure some test time options
    parser.add_argument("--oracle_aec", action="store_true")
    parser.add_argument("--no_aec", action="store_true")
    parser.add_argument("--real_data", action="store_true")
    parser.add_argument("--kws_loc", type=str, default=None)
    parser.add_argument("--kws_mode", type=str, default=None)
    parser.add_argument("--file_len", type=int, default=64000)

    # decide what to save
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
    system, kwargs, outer_learnable = get_system_ckpt(
        ckpt_loc, epoch, eval_kwargs["kws_loc"], eval_kwargs["kws_mode"]
    )
    fit_infer = system.make_fit_infer(outer_learnable=outer_learnable)

    # build the outputs path
    out_dir = os.path.join(
        eval_kwargs["out_dir"],
        eval_kwargs["name"],
        eval_kwargs["date"],
        f"epoch_{epoch}",
    )
    if eval_kwargs["oracle_aec"]:
        kws_mode = eval_kwargs["kws_mode"]
        out_dir = os.path.join(
            eval_kwargs["out_dir"],
            f"oracle_aec_{kws_mode}",
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "epoch_0",
        )
    elif eval_kwargs["no_aec"] and "no_aec" not in eval_kwargs["name"]:
        kws_mode = eval_kwargs["kws_mode"]
        out_dir = os.path.join(
            eval_kwargs["out_dir"],
            f"no_aec_{kws_mode}",
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            "epoch_0",
        )

    if eval_kwargs["save_outputs"] or eval_kwargs["save_metrics"]:
        os.makedirs(out_dir, exist_ok=True)

    test_dataset = KWSAECDataset(
        max_len=eval_kwargs["file_len"], mode="test", kws_mode=kwargs["kws_mode"]
    )

    if eval_kwargs["real_data"]:
        test_dataset = RealKWSAECDataset(
            max_len=eval_kwargs["file_len"], mode="test", kws_mode=kwargs["kws_mode"]
        )

    kws_params, kws_apply = load_kws_model(kwargs["kws_loc"])
    if kwargs["joint_train_kws"]:
        kws_params = outer_learnable["kws_p"]

    test_batch_size = 32
    test_loader = NumpyLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=2,
    )

    fs = 16000
    all_metrics = []
    for i, data in enumerate(tqdm.tqdm(test_loader)):
        if eval_kwargs["oracle_aec"]:
            aec_pred = data["signals"]["s"]
        elif eval_kwargs["no_aec"]:
            aec_pred = data["signals"]["d"]
        else:
            aec_pred = system.infer(data, fit_infer=fit_infer)[0]

        for j in range(aec_pred.shape[0]):
            kws_scores = kws_apply(kws_params, None, aec_pred[j])
            kws_pred = jnp.argmax(kws_scores, axis=-1)

            cur_metrics = {}
            cur_metrics["kws_scores"] = np.array(kws_scores)
            cur_metrics["kws_pred"] = kws_pred
            cur_metrics["kws_label"] = data["metadata"]["label"][j]
            cur_metrics["kws_acc"] = kws_pred == data["metadata"]["label"][[j]]

            other_metrics = get_all_metrics(
                data["signals"]["s"][j, :, 0],
                aec_pred[j, :, 0],
                data["signals"]["e"][j, :, 0],
                data["signals"]["d"][j, :, 0],
                fs=fs,
            )
            cur_metrics["erle"] = other_metrics["erle"]
            all_metrics.append(cur_metrics)

            # possibly save the outputs
            if eval_kwargs["save_outputs"]:
                sf.write(
                    os.path.join(out_dir, f"{test_batch_size*i+j}_out.wav"),
                    aec_pred[j, :, 0],
                    fs,
                )

    # print metrics
    mean_metrics = metrics.get_mean_metrics(all_metrics)
    std_metrics = metrics.get_std_metrics(all_metrics)

    for k in mean_metrics.keys():
        a_mean, a_sd = mean_metrics[k], std_metrics[k]
        print(f"{k}:{a_mean:.3f}+-{a_sd:.3f}")

    # if saving outputs save all the metrics
    if eval_kwargs["save_metrics"]:
        os.makedirs(out_dir, exist_ok=True)
        kws_mode = eval_kwargs["kws_mode"]
        real = eval_kwargs["real_data"]
        use_kws_ckpt = eval_kwargs["use_kws_ckpt"]

        if eval_kwargs["file_len"] == 64000:
            metrics_dir = os.path.join(
                out_dir, f"{kws_mode}_real_{real}_kwsckpt_{use_kws_ckpt}_metrics.pkl"
            )
        else:
            file_len = eval_kwargs["file_len"]
            metrics_dir = os.path.join(
                out_dir,
                f"{kws_mode}_real_{real}_kwsckpt_{use_kws_ckpt}_{file_len}_metrics.pkl",
            )
        with open(metrics_dir, "wb") as f:
            pickle.dump(all_metrics, f)
