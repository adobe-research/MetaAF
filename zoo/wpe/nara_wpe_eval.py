import argparse
import pickle
import os
import pprint
import tqdm
import soundfile as sf
import numpy as np
import datetime

"""
The code for running NARA and its setup are modified from
https://github.com/fgnt/nara_wpe/blob/master/examples/WPE_Numpy_online.ipynb
"""
from nara_wpe.wpe import online_wpe_step, get_power_online, OnlineWPE
from nara_wpe.utils import stft, istft, get_stft_center_frequencies

import zoo.wpe.wpe as wpe
from zoo.wpe.wpe_eval import get_all_metrics
from zoo import metrics


def fit_nara(y, kwargs):
    # grab all the nara params
    size = kwargs["window_size"]
    shift = kwargs["hop_size"]
    taps = kwargs["n_taps"]
    delay = kwargs["delay"]
    alpha = kwargs["alpha"]
    channel = kwargs["n_in_chan"]
    frequency_bins = size // 2 + 1

    Y = stft(y.T, size=size, shift=shift).transpose(1, 2, 0)
    T, _, _ = Y.shape

    def aquire_framebuffer():
        # buffer init with zeros so output is time alligned
        buffer = list(np.zeros((taps + delay, Y.shape[1], Y.shape[2]), dtype=Y.dtype))
        # buffer = list(Y[: taps + delay, :, :])
        # for t in range(taps + delay + 1, T):

        for t in range(0, T):

            buffer.append(Y[t, :, :])
            yield np.array(buffer)
            buffer.pop(0)

    Z_list = []
    online_wpe = OnlineWPE(
        taps=taps,
        delay=delay,
        alpha=alpha,
        channel=channel,
        frequency_bins=frequency_bins,
    )
    for Y_step in aquire_framebuffer():
        Z_list.append(online_wpe.step_frame(Y_step))

    Z = np.stack(Z_list)
    z = istft(
        np.asarray(Z).transpose(2, 0, 1),
        size=size,
        shift=shift,
    )

    # return the output for the first channel
    return z[0]


if __name__ == "__main__":

    # use same ckpt description as other models
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")

    # actual nara parameters
    parser.add_argument("--n_in_chan", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--hop_size", type=int, default=256)
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--n_taps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.9999)

    parser.add_argument("--out_dir", type=str, default="./outputs")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_metrics", action="store_true")

    eval_kwargs = vars(parser.parse_args())
    eval_kwargs["date"] = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fs = 16000
    epoch_num = 0
    pprint.pprint(eval_kwargs)

    # build the outputs path
    if eval_kwargs["save_outputs"] or eval_kwargs["save_metrics"]:
        out_dir = os.path.join(
            eval_kwargs["out_dir"],
            eval_kwargs["name"],
            eval_kwargs["date"],
            f"epoch_{epoch_num}",
        )
        os.makedirs(out_dir, exist_ok=True)

    # evaluate the model
    test_dataset = wpe.ReverbDataset(mode="test", n_mics=eval_kwargs["n_in_chan"])
    all_metrics = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        u, d = data["u"], data["d"]
        pred = fit_nara(d, eval_kwargs)

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
