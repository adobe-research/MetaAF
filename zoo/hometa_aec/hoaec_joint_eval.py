import argparse
import pickle
import os
import json
import pprint
import tqdm
import soundfile as sf
from scipy.io import wavfile
import numpy as np

from metaaf import optimizer_hogru
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf.optimizer_hogru import HOElementWiseGRU, Identity

import hometa_aec.hoaec_joint as aec
from zoo import metrics

from hometa_aec.hoaec import (
    MSFTAECDataset,
    AECOLS,
    _AECOLS_fwd,
    aec_loss,
)

from __config__ import AEC_DATA_DIR, RES_DATA_DIR, RES_US_DATA_DIR


def get_all_metrics(clean, enhanced, echo, mix, fs=16000):
    l = min(len(clean), len(enhanced), len(mix))
    clean, enhanced, mix = clean[:l], enhanced[:l], mix[:l]

    res = {}
    res["stoi"] = metrics.stoi(enhanced, clean, fs=fs)
    res["sisdr"] = metrics.sisdr(enhanced, clean)
    res["snr"] = metrics.snr(enhanced, clean)
    res["ssnr"] = metrics.snr(enhanced, clean, segmental=True)

    in_res = {}
    in_res["stoi"] = metrics.stoi(mix, clean, fs=fs)
    in_res["sisdr"] = metrics.sisdr(mix, clean)
    in_res["snr"] = metrics.snr(mix, clean)
    in_res["ssnr"] = metrics.snr(mix, clean, segmental=True)

    delta_res = {f"delta_{k}": res[k] - in_res[k] for k in in_res}
    in_res = {f"in_{k}": in_res[k] for k in in_res}

    res.update(delta_res)
    res.update(in_res)
    return res


def get_system_ckpt(res_ckpt_dir, res_e, mode):
    res_ckpt_loc = os.path.join(res_ckpt_dir, f"epoch_{res_e}.pkl")
    with open(res_ckpt_loc, "rb") as f:
        res_outer_learnable = pickle.load(f)

    res_kwargs_loc = os.path.join(res_ckpt_dir, "all_kwargs.json")
    with open(res_kwargs_loc, "rb") as f:
        res_kwargs = json.load(f)
    
    if mode == "res":
        kwargs = res_kwargs
        outer_learnable = res_outer_learnable

    pprint.pprint(kwargs)

    train_loader = NumpyLoader(MSFTAECDataset(mode="train"), batch_size=1)

    meta_sup_mse_loss = aec.make_meta_sup_mse_loss(kwargs["m_window_size"], kwargs["m_hop_size"], False, True)
    neg_sisdr_val = aec.make_neg_sisdr_val(kwargs["m_window_size"], kwargs["m_hop_size"])

    if mode == "res":
        _filter_fwd=aec._AECIdentity_fwd
        filter_kwargs=aec.AECIdentity.grab_args(kwargs)
        _optimizer_fwd = optimizer_hogru._identity_fwd
        optimizer_kwargs=Identity.grab_args(kwargs)
        init_optimizer=optimizer_hogru.init_optimizer_identity
        make_mapped_optmizer=optimizer_hogru.make_mapped_optmizer_identity
        _postprocess_fwd=aec._gru_ola_fwd

    system = MetaAFTrainer(
        _filter_fwd=_filter_fwd,
        filter_kwargs=filter_kwargs,
        filter_loss=aec_loss,
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
        meta_train_loss=meta_sup_mse_loss,
        meta_val_loss=neg_sisdr_val,
        _optimizer_fwd=_optimizer_fwd,
        optimizer_kwargs=optimizer_kwargs,
        init_optimizer=init_optimizer,
        make_mapped_optmizer=make_mapped_optmizer,
        _postprocess_fwd=_postprocess_fwd,
        postprocess_kwargs=aec.MaskerGRUOLA.grab_args(kwargs),
        kwargs=kwargs,
    )

    system.outer_learnable = outer_learnable
    return system, kwargs, outer_learnable

"""
python hoaec_joint_eval.py --mode res --name aec_res_banded --date 2022_05_24_23_06_49 --epoch 98 --aec_name aec_32_banded --save_metrics --save_outputs

"""


if __name__ == "__main__":
    
    """
        - eval_mode:
            - aec-res: joint evaluation on original MSFT Dataset, not supported
            - res: dnn-res evaluation on RES Dataset, need aec name, if aec name is none, eval gru on msft set
    """
    
    
    # get checkpoint description from user
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="res")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0)

    parser.add_argument("--aec_name", type=str, default="")
    
    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_metrics", action="store_true")


    eval_kwargs = vars(parser.parse_args())
    
    assert eval_kwargs["mode"] in ["res"]

    pprint.pprint(eval_kwargs)

    # build the checkpoint path
    res_ckpt_loc = os.path.join("./ckpts", eval_kwargs["name"], eval_kwargs["date"])
    res_epoch = int(eval_kwargs["epoch"])

    # load the checkpoint and kwargs file
    system, kwargs, outer_learnable = get_system_ckpt(
        res_ckpt_loc, res_epoch, eval_kwargs["mode"]
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
        
        
    if eval_kwargs["aec_name"] == "none":
        test_loader = NumpyLoader(MSFTAECDataset(mode="test", 
                                                 double_talk=kwargs["double_talk"],
                                                 random_roll=kwargs["random_roll"],
                                                 denoising=kwargs["denoising"]), 
                                  batch_size=16,
                                  num_workers=2,)

    else:
        test_loader = NumpyLoader(aec.RESDataset(aec_name=eval_kwargs["aec_name"], 
                                                 mode="test", 
                                                 random_roll=True,), 
                                  batch_size=16, 
                                  num_workers=2,)

    all_metrics = []

    for i, data in enumerate(tqdm.tqdm(test_loader)):
        preds = system.infer(data, fit_infer=fit_infer)[0]

        for j in range(preds.shape[0]):
            pred = preds[j,256:,0]
            u = data["signals"]["u"][j,:,0]
            d = data["signals"]["d"][j,:,0]
            e = data["signals"]["e"][j,:,0]
            s = data["signals"]["s"][j,:,0]

            l = min(len(s), len(pred), len(d), len(e), len(u))

            pred, d, s, e, u = pred[:l], d[:l], s[:l], e[:l], u[:l]
            all_metrics.append(get_all_metrics(s, pred, e, d, fs=fs))

            # possibly save the outputs
            if eval_kwargs["save_outputs"]:
                sf.write(os.path.join(out_dir, f"{16*i+j}_out.wav"), pred, fs)

    # print metrics
    mean_metrics = metrics.get_mean_metrics(all_metrics)
    std_metrics = metrics.get_std_metrics(all_metrics)

    for k in mean_metrics.keys():
        a_mean, a_sd = mean_metrics[k], std_metrics[k]
        print(f"{k}:{a_mean:.3f}+-{a_sd:.3f}")

    #metric_list = [all_metrics, dt_metrics, sd1_metrics, sd2_metrics, s1_metrics, s2_metrics]
    # if saving outputs save all the metrics
    if eval_kwargs["save_metrics"]:
        metrics_pkl = os.path.join(out_dir, "metrics.pkl")
        with open(metrics_pkl, "wb") as f:
            pickle.dump(all_metrics, f)