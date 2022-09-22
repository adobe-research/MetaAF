import argparse
import pickle
import os
import json
import pprint
import tqdm
import soundfile as sf
import numpy as np
import haiku as hk
import jax.numpy as jnp

import metaaf
from metaaf.data import NumpyLoader
from metaaf.meta import MetaAFTrainer
from metaaf import optimizer_hogru
from metaaf.optimizer_hogru import HOElementWiseGRU
from metaaf.filter import OverlapSave

from zoo import metrics
from zoo.__config__ import AEC_DATA_DIR, RES_DATA_DIR
from zoo.aec.aec import MSFTAECDataset, AECOLS, _AECOLS_fwd, aec_loss, meta_log_mse_loss, neg_erle_val_loss


def get_all_metrics(clean, enhanced, echo, mix, fs=16000):
    l = min(len(clean), len(enhanced), len(mix))
    clean, enhanced, mix = clean[:l], enhanced[:l], mix[:l]

    res = {}
    res["stoi"] = metrics.stoi(enhanced, clean, fs=fs)
    res["erle"] = metrics.erle(enhanced, mix, echo)
    res["serle"] = metrics.erle(enhanced, mix, echo, segmental=True, window_size=4096, hop_size=2048)
    
    res["sisdr"] = metrics.sisdr(enhanced, clean)
    res["snr"] = metrics.snr(enhanced, clean)

    in_res = {}
    in_res["stoi"] = metrics.stoi(mix, clean, fs=fs)
    in_res["erle"] = metrics.erle(mix, mix, echo)
    in_res["serle"] = metrics.erle(mix, mix, echo, segmental=True, window_size=4096, hop_size=2048)
    
    in_res["sisdr"] = metrics.sisdr(mix, clean)
    in_res["snr"] = metrics.snr(mix, clean)

    delta_res = {f"delta_{k}": res[k] - in_res[k] for k in in_res}
    in_res = {f"in_{k}": in_res[k] for k in in_res}

    res.update(delta_res)
    res.update(in_res)
    return res


def get_system_ckpt(ckpt_dir, e, model_type="egru", no_extra=False, iwaenc_release=False):
    ckpt_loc = os.path.join(ckpt_dir, f"epoch_{e}.pkl")
    with open(ckpt_loc, "rb") as f:
        outer_learnable = pickle.load(f)

    kwargs_loc = os.path.join(ckpt_dir, "all_kwargs.json")
    with open(kwargs_loc, "rb") as f:
        kwargs = json.load(f)

    pprint.pprint(kwargs)

    train_loader = NumpyLoader(MSFTAECDataset(mode="train"), batch_size=1)

    meta_train_loss = meta_log_mse_loss

    optimizer_kwargs = HOElementWiseGRU.grab_args(kwargs)
    _optimizer_fwd = optimizer_hogru._elementwise_hogru_fwd
    
    if no_extra:
        init_optimizer = optimizer_hogru.init_optimizer
        make_mapped_optmizer=optimizer_hogru.make_mapped_optmizer
    else:
        if iwaenc_release:
            init_optimizer = optimizer_hogru.init_optimizer_all_data
            make_mapped_optmizer=optimizer_hogru.make_mapped_optmizer_all_data_iwaenc
        else:
            init_optimizer = optimizer_hogru.init_optimizer_all_data
            make_mapped_optmizer=optimizer_hogru.make_mapped_optmizer_all_data
        
    if iwaenc_release:
        _filter_fwd = _AECOLS_IWAENC_fwd
        filter_kwargs = AECOLS_IWAENC.grab_args(kwargs)
    else:
        _filter_fwd = _AECOLS_fwd
        filter_kwargs = AECOLS.grab_args(kwargs)
    
    system = MetaAFTrainer(
        _filter_fwd=_filter_fwd,
        filter_kwargs=filter_kwargs,
        filter_loss=aec_loss,
        optimizer_kwargs=optimizer_kwargs,
        train_loader=train_loader,
        val_loader=train_loader,
        test_loader=train_loader,
        meta_train_loss=meta_train_loss,
        meta_val_loss=neg_erle_val_loss,
        _optimizer_fwd=_optimizer_fwd,
        init_optimizer=init_optimizer,
        make_mapped_optmizer=make_mapped_optmizer,
        kwargs=kwargs,
    )
    system.outer_learnable = outer_learnable
    return system, kwargs, outer_learnable


class AECOLS_IWAENC(OverlapSave, hk.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # select the analysis window
        self.analysis_window = jnp.ones(self.window_size)

    def __ols_call__(self, u, d, metadata):
        w = self.get_filter(name="w")
        
        y = (w * u).sum(0)
        out = d[-1] - y
        
        return {
            "out": out,
            "u": u,
            "d": d[-1, None],
            "y": y[None],
            "e": out[None],
            "loss": jnp.vdot(out, out).real / out.size,
        }

    @staticmethod
    def add_args(parent_parser):
        return super(AECOLS, AECOLS).add_args(parent_parser)

    @staticmethod
    def grab_args(kwargs):
        return super(AECOLS, AECOLS).grab_args(kwargs)


def _AECOLS_IWAENC_fwd(u, d, e, s, metadata=None, init_data=None, **kwargs):
    gen_filter = AECOLS_IWAENC(**kwargs)
    return gen_filter(u=u, d=d)


if __name__ == "__main__":
    
    """
    # CPU EVAL
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from jax import config
    config.update('jax_platform_name', 'cpu')
    """
    
    # get checkpoint description from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--model_type", type=str, default="egru")

    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_metrics", action="store_true")
    parser.add_argument("--scene_change", action="store_true")
    parser.add_argument("--no_extra", action="store_true")
    parser.add_argument("--fix_train_roll", action="store_true")
    
    parser.add_argument("--generate_aec_data", action="store_true")
    
    parser.add_argument("--iwaenc_release", action="store_true")

    eval_kwargs = vars(parser.parse_args())
    pprint.pprint(eval_kwargs)

    # build the checkpoint path
    ckpt_loc = os.path.join("./ckpts", eval_kwargs["name"], eval_kwargs["date"])
    epoch = int(eval_kwargs["epoch"])

    # load the checkpoint and kwargs file
    system, kwargs, outer_learnable = get_system_ckpt(
        ckpt_loc, epoch, model_type=eval_kwargs["model_type"], 
        no_extra = eval_kwargs["no_extra"], iwaenc_release=eval_kwargs["iwaenc_release"]
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
        
    # generate aec training data for DNN-RES
    if eval_kwargs["generate_aec_data"]:
        all_metrics = []
        
        generate_dir = os.path.join(RES_DATA_DIR, eval_kwargs["name"])
        if os.path.isdir(generate_dir):
            print("AEC outputs directory exists, skip")
        
        else:
            print("Storing AEC outputs...")
            os.makedirs(generate_dir)

            # nums = [0,9000,9500]
            nums = {"train":1000, "val":500, "test":0}

            for j, mode in enumerate(["test", "val", "train"]):

                # random_roll = kwargs["random_roll"]

                data_loader = NumpyLoader(
                    MSFTAECDataset(mode=mode, 
                                   double_talk=kwargs["double_talk"], 
                                   random_roll=kwargs["random_roll"],
                                   fix_train_roll=eval_kwargs["fix_train_roll"]),
                    batch_size=16,
                    shuffle=False,
                    num_workers=10,
                )

                for i, data in enumerate(tqdm.tqdm(data_loader)):
                    preds = system.infer(data)[0]

                    for k in range(preds.shape[0]):
                        pred = preds[k,:,0]
                        u = data["signals"]["u"][k,:,0]
                        d = data["signals"]["d"][k,:,0]
                        e = data["signals"]["e"][k,:,0]
                        s = data["signals"]["s"][k,:,0]

                        l = min(len(s), len(pred), len(d), len(e), len(u))

                        pred, d, s, e, u = pred[:l], d[:l], s[:l], e[:l], u[:l]
                        all_metrics.append(get_all_metrics(s, pred, e, d, fs=fs))

                        sf.write(os.path.join(generate_dir, f"{16*i + k + nums[mode]}_out.wav"), pred, fs)
                        # sf.write(os.path.join(generate_dir, f"{16*i + k + nums[mode]}_u.wav"), u, fs)
                        # sf.write(os.path.join(generate_dir, f"{16*i + k + nums[mode]}_s.wav"), s, fs)

            mean_metrics = metrics.get_mean_metrics(all_metrics)
            std_metrics = metrics.get_std_metrics(all_metrics)

            for k in mean_metrics.keys():
                a_mean, a_sd = mean_metrics[k], std_metrics[k]
                print(f"{k}:{a_mean:.3f}+-{a_sd:.3f}")
            
            
    else:
        
        test_loader = NumpyLoader(
            MSFTAECDataset(mode="test", 
                           double_talk=kwargs["double_talk"], 
                           random_roll=kwargs["random_roll"]),
            batch_size=16,
            shuffle=False,
            num_workers=2,
        )

        all_metrics = []

        for i, data in enumerate(tqdm.tqdm(test_loader)):
            preds = system.infer(data)[0]

            for j in range(preds.shape[0]):
                pred = preds[j,:,0]
                u = data["signals"]["u"][j,:,0]
                d = data["signals"]["d"][j,:,0]
                e = data["signals"]["e"][j,:,0]
                s = data["signals"]["s"][j,:,0]

                l = min(len(s), len(pred), len(d), len(e), len(u))

                pred, d, s, e, u = pred[:l], d[:l], s[:l], e[:l], u[:l]
                all_metrics.append(get_all_metrics(s, pred, e, d, fs=fs))

                if eval_kwargs["save_outputs"]:
                    sf.write(os.path.join(out_dir, f"{16*i+j}_out.wav"), pred, fs)

        # print metrics
        mean_metrics = metrics.get_mean_metrics(all_metrics)
        std_metrics = metrics.get_std_metrics(all_metrics)

        for k in mean_metrics.keys():
            a_mean, a_sd = mean_metrics[k], std_metrics[k]
            print(f"{k}:{a_mean:.3f}+-{a_sd:.3f}")

        # if saving outputs save all the metrics
        if eval_kwargs["save_metrics"]:
            metrics_pkl = os.path.join(out_dir, "metrics.pkl")
            if eval_kwargs["scene_change"]:
                metrics_pkl = os.path.join(out_dir, "metrics_scene_change.pkl")
            with open(metrics_pkl, "wb") as f:
                pickle.dump(all_metrics, f)
