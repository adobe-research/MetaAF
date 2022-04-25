import numpy as np
import pystoi
from pysepm import fwSNRseg

EPS = 1e-10


def get_std_metrics(metrics):
    means = {}
    for k in metrics[0].keys():
        means[k] = np.nanstd([m[k] for m in metrics])
    return means


def get_mean_metrics(metrics):
    means = {}
    for k in metrics[0].keys():
        means[k] = np.nanmean([m[k] for m in metrics])
    return means


def stoi(out_sig, target_sig, fs=16000):
    min_len = min(len(out_sig), len(target_sig))
    out_sig, target_sig = out_sig[:min_len], target_sig[:min_len]

    return pystoi.stoi(target_sig, out_sig, fs)


def fwssnr(out_sig, target_sig, fs=16000):
    min_len = min(len(out_sig), len(target_sig))
    out_sig, target_sig = out_sig[:min_len], target_sig[:min_len]

    return fwSNRseg(target_sig, out_sig, fs)


def snr(out_sig, target_sig, segmental=False, window_size=1024, hop_size=512):
    min_len = min(len(out_sig), len(target_sig))
    out_sig = out_sig[:min_len]
    target_sig = target_sig[:min_len]
    res_sig = target_sig - out_sig

    if segmental:
        res_chunk = np.array(
            [
                res_sig[i : i + window_size]
                for i in range(0, len(res_sig) - window_size, hop_size)
            ]
        )

        target_chunk = np.array(
            [
                target_sig[i : i + window_size]
                for i in range(0, len(target_sig) - window_size, hop_size)
            ]
        )

        ssnr = 10 * np.log10(
            np.mean(np.abs(target_chunk) ** 2, axis=1)
            / (np.mean(np.abs(res_chunk) ** 2, axis=1) + EPS)
            + EPS
        )

        return ssnr
    else:
        snr = 10 * np.log10(
            np.mean(np.abs(target_sig) ** 2) / (np.mean(np.abs(res_sig) ** 2) + EPS)
            + EPS
        )
        return snr


def erle(out_sig, ref_sig, target_sig, segmental=False, window_size=1024, hop_size=512):
    min_len = min(len(out_sig), len(ref_sig), len(target_sig))
    out_sig = out_sig[:min_len]
    ref_sig = ref_sig[:min_len]
    target_sig = target_sig[:min_len]

    res_sig = target_sig - (ref_sig - out_sig)

    if segmental:
        res_chunk = np.array(
            [
                res_sig[i : i + window_size]
                for i in range(0, len(res_sig) - window_size, hop_size)
            ]
        )

        target_chunk = np.array(
            [
                target_sig[i : i + window_size]
                for i in range(0, len(target_sig) - window_size, hop_size)
            ]
        )

        serle = 10 * np.log10(
            np.mean(np.abs(target_chunk) ** 2, axis=1)
            / (np.mean(np.abs(res_chunk) ** 2, axis=1) + EPS)
            + EPS
        )

        return serle
    else:
        erle = 10 * np.log10(
            np.mean(np.abs(target_sig) ** 2) / (np.mean(np.abs(res_sig) ** 2) + EPS)
            + EPS
        )
        return erle


def sisdr(out_sig, target_sig):
    min_len = min(len(out_sig), len(target_sig))
    out_sig, target_sig = out_sig[:min_len], target_sig[:min_len]

    tt = np.dot(target_sig, target_sig)
    ot = np.dot(target_sig, out_sig)

    t_proj = (ot + EPS) / (tt + EPS) * target_sig
    res = out_sig - t_proj

    return 10 * np.log10(((t_proj ** 2).mean() + EPS) / ((res ** 2).mean() + EPS) + EPS)


def srr_stft(out_sig, in_sig, segmental=False, window_size=1024, hop_size=512):
    min_len = min(len(out_sig), len(in_sig))
    out_sig, in_sig = out_sig[:min_len], in_sig[:min_len]

    out_stft = np.fft.rfft(
        np.array(
            [
                out_sig[i : i + window_size]
                for i in range(0, len(out_sig) - window_size, hop_size)
            ]
        ),
        axis=1,
    )

    in_stft = np.fft.rfft(
        np.array(
            [
                in_sig[i : i + window_size]
                for i in range(0, len(in_sig) - window_size, hop_size)
            ]
        ),
        axis=1,
    )

    srr_signal = 10 * np.log10(
        np.mean(np.abs(out_stft) ** 2, axis=1)
        / (np.mean(np.abs(in_stft - out_stft) ** 2, axis=1) + EPS)
        + EPS
    )

    if segmental:
        return srr_signal
    else:
        return srr_signal.mean()
