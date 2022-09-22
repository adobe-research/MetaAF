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


def run_speex(u, d, fs=16000, system_len=2048, frame_size=512):
    """Acoustic Echo Cancellation for wav files."""

    # write the nearend and farend arrays
    tempdir = tempfile.mkdtemp()
    farend_path = os.path.join(tempdir, "farend.wav")
    sf.write(farend_path, u.flatten(), fs)

    nearend_path = os.path.join(tempdir, "nearend.wav")
    sf.write(nearend_path, d.flatten(), fs)

    out_path = os.path.join(tempdir, "output.wav")

    near = wave.open(nearend_path, "rb")
    far = wave.open(farend_path, "rb")

    if near.getnchannels() > 1 or far.getnchannels() > 1:
        print("Only support mono channel")
        sys.exit(2)

    out = wave.open(out_path, "wb")
    out.setnchannels(near.getnchannels())
    out.setsampwidth(near.getsampwidth())
    out.setframerate(near.getframerate())

    echo_canceller = EchoCanceller.create(frame_size, system_len, near.getframerate())

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


def run_eval(test_dataset, out_dir, test_dataset_name, eval_kwargs, fs=16000):
    all_metrics = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        data = test_dataset[i]
        u, d, e, s = (
            data["signals"]["u"],
            data["signals"]["d"],
            data["signals"]["e"],
            data["signals"]["s"],
        )
        pred = run_speex(u, d, fs=fs, system_len=eval_kwargs["system_len"])

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


"""
Universal speex
python speex_aec_eval.py --name aec_combo_speex --save_metrics
"""

if __name__ == "__main__":

    # get checkpoint description from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")

    parser.add_argument("--out_dir", type=str, default="./taslp_rebut_outputs")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--save_metrics", action="store_true")
    parser.add_argument("--system_len", type=int, default=2048)

    eval_kwargs = vars(parser.parse_args())
    eval_kwargs["date"] = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fs = 16000
    epoch = 0
    pprint.pprint(eval_kwargs)

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
    true_rir_len = "DEFAULT"
    system_len = "DEFAULT"

    test_datasets = [
        aec.MSFTAECDataset_RIR(
            mode="test",
            double_talk=False,
            scene_change=False,
            random_roll=True,
            random_level=False,
        ),
        aec.MSFTAECDataset_RIR(
            mode="test",
            double_talk=True,
            scene_change=False,
            random_roll=True,
            random_level=False,
        ),
        aec.MSFTAECDataset_RIR(
            mode="test",
            double_talk=True,
            scene_change=True,
            random_roll=True,
            random_level=False,
        ),
        aec.MSFTAECDataset(
            mode="test",
            double_talk=True,
            scene_change=False,
            random_roll=True,
            random_level=False,
        ),
        aec.MSFTAECDataset(
            mode="test",
            double_talk=True,
            scene_change=True,
            random_roll=True,
            random_level=False,
        ),
    ]

    test_dataset_names = [
        f"DT_{False}_SC_{False}_NL_{False}_TR_{true_rir_len}_SL_{system_len}",
        f"DT_{True}_SC_{False}_NL_{False}_TR_{true_rir_len}_SL_{system_len}",
        f"DT_{True}_SC_{True}_NL_{False}_TR_{true_rir_len}_SL_{system_len}",
        f"DT_{True}_SC_{False}_NL_{True}_TR_{true_rir_len}_SL_{system_len}",
        f"DT_{True}_SC_{True}_NL_{True}_TR_{true_rir_len}_SL_{system_len}",
    ]

    for i, test_dataset in enumerate(test_datasets):
        print(" --- Testing Dataset ---")
        print(f" --- {test_dataset_names[i]} ---")

        all_metrics = run_eval(
            test_dataset, out_dir, test_dataset_names[i], eval_kwargs
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
