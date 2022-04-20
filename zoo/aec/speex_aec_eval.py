import argparse
import pickle
import os
import pprint
import tqdm
import soundfile as sf
import numpy as np
import datetime
import tempfile
import sys
import shutil
import wave

from speexdsp import EchoCanceller

import zoo.aec.aec as aec
from zoo.aec.aec_eval import get_all_metrics
from zoo import metrics


def run_speex(u, d, fs=16000):
    """Acoustic Echo Cancellation for wav files."""

    # write the nearend and farend arrays
    tempdir = tempfile.mkdtemp()
    farend_path = os.path.join(tempdir, "farend.wav")
    sf.write(farend_path, u.flatten(), fs)

    nearend_path = os.path.join(tempdir, "nearend.wav")
    sf.write(nearend_path, d.flatten(), fs)

    out_path = os.path.join(tempdir, "output.wav")

    frame_size = 1024
    near = wave.open(nearend_path, "rb")
    far = wave.open(farend_path, "rb")

    if near.getnchannels() > 1 or far.getnchannels() > 1:
        print("Only support mono channel")
        sys.exit(2)

    out = wave.open(out_path, "wb")
    out.setnchannels(near.getnchannels())
    out.setsampwidth(near.getsampwidth())
    out.setframerate(near.getframerate())

    echo_canceller = EchoCanceller.create(frame_size, 1024, near.getframerate())

    in_data_len = frame_size
    in_data_bytes = frame_size * 2
    out_data_len = frame_size
    out_data_bytes = frame_size * 2

    while True:
        in_data = near.readframes(in_data_len)
        out_data = far.readframes(out_data_len)
        if len(in_data) != in_data_bytes or len(out_data) != out_data_bytes:
            break
        in_data = echo_canceller.process(in_data, out_data)
        out.writeframes(in_data)

    near.close()
    far.close()
    out.close()

    output = sf.read(out_path)[0]
    shutil.rmtree(tempdir)
    return output


"""
Single Talk
python speex_aec_eval.py --name aec_st_speex --random_roll

Double Talk
python speex_aec_eval.py --name aec_dt_speex --random_roll --double_talk

Double Talk with Scene Change
python speex_aec_eval.py --name aec_dt_sc_speex --random_roll --double_talk --scene_change

Double Talk Nonlinear
python speex_aec_eval.py --name aec_dt_nl_speex --random_roll --double_talk --dataset nonlinear
"""

if __name__ == "__main__":

    # get checkpoint description from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")

    parser.add_argument("--out_dir", type=str, default="./v2_outputs")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_metrics", action="store_true")

    parser.add_argument("--scene_change", action="store_true")
    parser.add_argument("--double_talk", action="store_true")
    parser.add_argument("--random_roll", action="store_true")
    parser.add_argument("--repeat_scene", action="store_true")
    parser.add_argument("--dataset", type=str, default="linear")

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

    if eval_kwargs["dataset"] == "linear":
        aec_dataset = aec.MSFTAECDataset_RIR
    elif eval_kwargs["dataset"] == "nonlinear":
        aec_dataset = aec.MSFTAECDataset

    # evaluate the model
    test_dataset = aec_dataset(
        mode="test",
        double_talk=eval_kwargs["double_talk"],
        scene_change=eval_kwargs["scene_change"],
        random_roll=eval_kwargs["random_roll"],
    )
    all_metrics = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        u, d, e, s = data["u"], data["d"], data["e"], data["s"]

        if eval_kwargs["repeat_scene"]:
            u = np.concatenate((u, u), axis=0)
            d = np.concatenate((d, d), axis=0)
            e = np.concatenate((e, e), axis=0)
            s = np.concatenate((s, s), axis=0)

        d_input = {"u": u[None], "d": d[None], "s": s[None], "e": e[None]}
        pred = run_speex(u, d, fs=fs)

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
        if eval_kwargs["scene_change"]:
            metrics_pkl = os.path.join(out_dir, "metrics_scene_change.pkl")
        elif eval_kwargs["repeat_scene"]:
            metrics_pkl = os.path.join(out_dir, "metrics_repeat.pkl")
        with open(metrics_pkl, "wb") as f:
            pickle.dump(all_metrics, f)
