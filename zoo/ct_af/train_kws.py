import numpy as np
import soundfile as sf
import pathlib
import pickle
import os
import glob2
import argparse
from torch.utils.data import Dataset
import haiku as hk
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp
import tqdm
from metaaf.data import NumpyLoader
from zoo.__config__ import AEC_DATA_DIR, KWS_DATA_DIR


class RealKWSAECDataset(Dataset):
    @staticmethod
    def collect_files(dir, valid_datanames, faulty_guids):
        files = []
        for valid_dataname in valid_datanames:
            query = dir + valid_dataname
            files.extend(glob2.glob(query))

        valid_files = []
        for file in files:
            is_faulty = False
            for faulty_guid in faulty_guids:
                is_faulty = faulty_guid in file

            if not is_faulty:
                valid_files.append(file)

        return valid_files

    def __init__(
        self,
        aec_dir=AEC_DATA_DIR,
        kws_dir=KWS_DATA_DIR,
        mode="train",
        aec_roll=True,
        kws_shift=True,
        max_len=160000,
        kws_mode="2cmds",
    ):
        self.max_len = max_len
        self.aec_roll = aec_roll
        self.kws_dir = kws_dir
        self.kws_shift = kws_shift
        self.kws_mode = kws_mode
        self.mode = mode
        self.min_ser = -25.0
        self.max_ser = 0.0

        # Setup the KWS portion of the dataset
        if self.kws_mode == "2cmds":
            self.kws_labels = [
                "left",
                "right",
            ]
        elif self.kws_mode == "10cmds":
            self.kws_labels = [
                "yes",
                "no",
                "up",
                "down",
                "left",
                "right",
                "on",
                "off",
                "stop",
                "go",
            ]
        elif self.kws_mode == "35cmds":
            self.kws_labels = [
                "backward",
                "bed",
                "bird",
                "cat",
                "dog",
                "down",
                "eight",
                "five",
                "follow",
                "forward",
                "four",
                "go",
                "happy",
                "house",
                "learn",
                "left",
                "marvin",
                "nine",
                "no",
                "off",
                "on",
                "one",
                "right",
                "seven",
                "sheila",
                "six",
                "stop",
                "three",
                "tree",
                "two",
                "up",
                "visual",
                "wow",
                "yes",
                "zero",
            ]

        self.kws_label_map = dict(zip(self.kws_labels, np.arange(len(self.kws_labels))))
        self.kws_one_hot_map = np.eye(len(self.kws_labels))

        self.kws_files = self.get_kws_files()
        if self.kws_shift:
            if mode == "val":
                np.random.seed(95)
                self.kws_shifts = np.random.randint(
                    0, self.max_len - 16000, size=len(self.kws_files)
                )
            elif mode == "test":
                np.random.seed(1337)
                self.kws_shifts = np.random.randint(
                    0, self.max_len - 16000, size=len(self.kws_files)
                )

        # Setup the AEC dataset
        self.farends = []
        self.nearends = []

        valid_datanames = [
            "/**_farend_singletalk_lpb.wav",
            "/**_farend_singletalk_with_movement_lpb.wav",
        ]

        faulty_guids = ["StfDere4UEKo86FFoQlU5Q"]

        if mode == "train":
            real_dir = os.path.join(aec_dir, "datasets/real")
            self.farends = self.collect_files(real_dir, valid_datanames, faulty_guids)
            self.nearends = [farend[:-7] + "mic.wav" for farend in self.farends]
        elif mode == "val":
            noisy_dir = os.path.join(aec_dir, "datasets/test_set/noisy/")
            clean_dir = os.path.join(aec_dir, "datasets/test_set/clean/")

            farends_noisy = self.collect_files(noisy_dir, valid_datanames, faulty_guids)
            farends_clean = self.collect_files(clean_dir, valid_datanames, faulty_guids)

            nearends_noisy = [farend[:-7] + "mic.wav" for farend in farends_noisy]
            nearends_clean = [farend[:-7] + "mic_c.wav" for farend in farends_clean]

            self.farends = farends_noisy + farends_clean
            self.nearends = nearends_noisy + nearends_clean
        else:
            noisy_dir = os.path.join(aec_dir, "datasets/blind_test_set/noisy/")
            clean_dir = os.path.join(aec_dir, "datasets/blind_test_set/clean/")

            farends_noisy = self.collect_files(noisy_dir, valid_datanames, faulty_guids)
            farends_clean = self.collect_files(clean_dir, valid_datanames, faulty_guids)

            nearends_noisy = [farend[:-7] + "mic.wav" for farend in farends_noisy]
            nearends_clean = [farend[:-7] + "mic.wav" for farend in farends_clean]

            self.farends = farends_noisy + farends_clean
            self.nearends = nearends_noisy + nearends_clean
        self.aec_len = len(self.farends)

        if self.aec_roll:
            if mode == "val":
                np.random.seed(95)
                self.aec_rolls = np.random.randint(
                    0, self.max_len, size=len(self.kws_files)
                )
            elif mode == "test":
                np.random.seed(1337)
                self.aec_rolls = np.random.randint(
                    0, self.max_len, size=len(self.kws_files)
                )

        # Setup the interface
        if mode == "val":
            np.random.seed(95)
            self.kws_aec_map = np.random.randint(
                0, len(self.farends), size=len(self.kws_files)
            )
            np.random.seed(95)
            self.sers = np.random.uniform(
                self.min_ser, self.max_ser, size=len(self.kws_files)
            )
        elif mode == "test":
            np.random.seed(1337)
            self.kws_aec_map = np.random.randint(
                0, len(self.farends), size=len(self.kws_files)
            )
            np.random.seed(1337)
            self.sers = np.random.uniform(
                self.min_ser, self.max_ser, size=len(self.kws_files)
            )

    def get_cmds(self, files):
        new_files = []
        for f in files:
            label = f.split("/")[0]
            if label in self.kws_labels:
                new_files.append(f)
        return new_files

    def get_kws_files(self):
        with open(os.path.join(self.kws_dir, "testing_list.txt"), "r") as f:
            test_files = f.readlines()
        test_files = [f[:-1] for f in test_files]

        with open(os.path.join(self.kws_dir, "validation_list.txt"), "r") as f:
            val_files = f.readlines()
        val_files = [f[:-1] for f in val_files]

        p = pathlib.Path(self.kws_dir)
        all_files = list(p.glob("**/*.wav"))
        all_files = [os.path.join(*f.parts[-2:]) for f in all_files]

        all_files = self.get_cmds(all_files)
        val_files = self.get_cmds(val_files)
        test_files = self.get_cmds(test_files)

        all_files_set = set(all_files)
        val_files_set = set(val_files)
        test_files_set = set(test_files)

        train_files_set = all_files_set - test_files_set.union(val_files_set)
        train_files = list(train_files_set)

        if self.mode == "test":
            return test_files
        elif self.mode == "val":
            return val_files
        elif self.mode == "train":
            return train_files

    def __len__(self):
        return len(self.kws_files)

    def __getitem__(self, idx):
        # get the word
        kws_file = self.kws_files[idx]
        kws_label = kws_file.split("/")[0]
        kws_digit = self.kws_label_map[kws_label]
        s, sr = sf.read(os.path.join(self.kws_dir, kws_file))
        s = np.pad(s, (0, max(0, self.max_len - len(s))))

        if self.kws_shift:
            shift = (
                np.random.randint(0, self.max_len - 16000)
                if self.mode == "train"
                else self.kws_shifts[idx]
            )
            s = np.roll(s, shift)

        # get the echo
        aec_idx = (
            np.random.randint(0, self.aec_len)
            if self.mode == "train"
            else self.kws_aec_map[idx]
        )

        u, sr = sf.read(self.farends[aec_idx])
        e, sr = sf.read(self.nearends[aec_idx])

        u = (
            np.pad(u, (0, max(0, self.max_len - len(u))))
            if len(u) < self.max_len
            else u[: self.max_len]
        )
        e = (
            np.pad(e, (0, max(0, self.max_len - len(e))))
            if len(e) < self.max_len
            else e[: self.max_len]
        )

        if self.aec_roll:
            shift = (
                np.random.randint(0, self.max_len)
                if self.mode == "train"
                else self.aec_rolls[idx]
            )
            u = np.roll(u, shift)
            e = np.roll(e, shift)

        ser = (
            np.random.uniform(self.min_ser, self.max_ser)
            if self.mode == "train"
            else self.sers[idx]
        )
        s_ser_scale = np.sqrt(
            np.abs(e**2).mean() * (10 ** (ser / 10)) / np.abs(s**2).mean()
        )
        s = s * s_ser_scale

        # mix the data
        d = s + e

        return {
            "signals": {
                "u": u[:, None],
                "d": d[:, None],
                "e": e[:, None],
                "s": s[:, None],
            },
            "metadata": {"label": kws_digit, "onehot": self.kws_one_hot_map[kws_digit]},
        }


class KWSAECDataset(Dataset):
    def __init__(
        self,
        aec_dir=AEC_DATA_DIR,
        kws_dir=KWS_DATA_DIR,
        mode="train",
        aec_roll=True,
        kws_shift=True,
        max_len=160000,
        kws_mode="2cmds",
    ):
        self.max_len = max_len
        self.aec_roll = aec_roll
        self.kws_dir = kws_dir
        self.kws_shift = kws_shift
        self.kws_mode = kws_mode
        self.mode = mode
        self.min_ser = -25.0
        self.max_ser = 0.0

        # Setup the KWS portion of the dataset
        if self.kws_mode == "2cmds":
            self.kws_labels = [
                "left",
                "right",
            ]
        elif self.kws_mode == "10cmds":
            self.kws_labels = [
                "yes",
                "no",
                "up",
                "down",
                "left",
                "right",
                "on",
                "off",
                "stop",
                "go",
            ]
        elif self.kws_mode == "35cmds":
            self.kws_labels = [
                "backward",
                "bed",
                "bird",
                "cat",
                "dog",
                "down",
                "eight",
                "five",
                "follow",
                "forward",
                "four",
                "go",
                "happy",
                "house",
                "learn",
                "left",
                "marvin",
                "nine",
                "no",
                "off",
                "on",
                "one",
                "right",
                "seven",
                "sheila",
                "six",
                "stop",
                "three",
                "tree",
                "two",
                "up",
                "visual",
                "wow",
                "yes",
                "zero",
            ]

        self.kws_label_map = dict(zip(self.kws_labels, np.arange(len(self.kws_labels))))
        self.kws_one_hot_map = np.eye(len(self.kws_labels))

        self.kws_files = self.get_kws_files()
        if self.kws_shift:
            if mode == "val":
                np.random.seed(95)
                self.kws_shifts = np.random.randint(
                    0, self.max_len - 16000, size=len(self.kws_files)
                )
            elif mode == "test":
                np.random.seed(1337)
                self.kws_shifts = np.random.randint(
                    0, self.max_len - 16000, size=len(self.kws_files)
                )

        # Setup the AEC dataset
        synthetic_dir = os.path.join(aec_dir, "datasets/synthetic/")
        self.echo_signal_dir = os.path.join(synthetic_dir, "echo_signal")
        self.farend_speech_dir = os.path.join(synthetic_dir, "farend_speech")
        self.nearend_mic_dir = os.path.join(synthetic_dir, "nearend_mic_signal")
        self.nearend_speech_dir = os.path.join(synthetic_dir, "nearend_speech")

        if self.aec_roll:
            if mode == "val":
                np.random.seed(95)
                self.aec_rolls = np.random.randint(
                    0, self.max_len, size=len(self.kws_files)
                )
            elif mode == "test":
                np.random.seed(1337)
                self.aec_rolls = np.random.randint(
                    0, self.max_len, size=len(self.kws_files)
                )

        if self.mode == "test":
            self.aec_offset = 0
            self.aec_len = 500
        elif self.mode == "val":
            self.aec_offset = 500
            self.aec_len = 500
        elif self.mode == "train":
            self.aec_offset = 1000
            self.aec_len = 9000

        # Setup the interface
        if mode == "val":
            np.random.seed(95)
            self.kws_aec_map = np.random.randint(0, 500, size=len(self.kws_files))
            np.random.seed(95)
            self.sers = np.random.uniform(
                self.min_ser, self.max_ser, size=len(self.kws_files)
            )
        elif mode == "test":
            np.random.seed(1337)
            self.kws_aec_map = np.random.randint(0, 500, size=len(self.kws_files))
            np.random.seed(1337)
            self.sers = np.random.uniform(
                self.min_ser, self.max_ser, size=len(self.kws_files)
            )

    def get_cmds(self, files):
        new_files = []
        for f in files:
            label = f.split("/")[0]
            if label in self.kws_labels:
                new_files.append(f)
        return new_files

    def get_kws_files(self):
        with open(os.path.join(self.kws_dir, "testing_list.txt"), "r") as f:
            test_files = f.readlines()
        test_files = [f[:-1] for f in test_files]

        with open(os.path.join(self.kws_dir, "validation_list.txt"), "r") as f:
            val_files = f.readlines()
        val_files = [f[:-1] for f in val_files]

        p = pathlib.Path(self.kws_dir)
        all_files = list(p.glob("**/*.wav"))
        all_files = [os.path.join(*f.parts[-2:]) for f in all_files]

        all_files = self.get_cmds(all_files)
        val_files = self.get_cmds(val_files)
        test_files = self.get_cmds(test_files)

        all_files_set = set(all_files)
        val_files_set = set(val_files)
        test_files_set = set(test_files)

        train_files_set = all_files_set - test_files_set.union(val_files_set)
        train_files = list(train_files_set)

        if self.mode == "test":
            return test_files
        elif self.mode == "val":
            return val_files
        elif self.mode == "train":
            return train_files

    def __len__(self):
        return len(self.kws_files)

    def __getitem__(self, idx):
        # get the word
        kws_file = self.kws_files[idx]
        kws_label = kws_file.split("/")[0]
        kws_digit = self.kws_label_map[kws_label]
        s, sr = sf.read(os.path.join(self.kws_dir, kws_file))
        s = np.pad(s, (0, max(0, self.max_len - len(s))))

        if self.kws_shift:
            shift = (
                np.random.randint(0, self.max_len - 16000)
                if self.mode == "train"
                else self.kws_shifts[idx]
            )
            s = np.roll(s, shift)

        # get the echo
        aec_idx = (
            np.random.randint(0, self.aec_len)
            if self.mode == "train"
            else self.kws_aec_map[idx]
        )
        aec_idx = aec_idx + self.aec_offset
        u, sr = sf.read(
            os.path.join(self.farend_speech_dir, f"farend_speech_fileid_{aec_idx}.wav")
        )

        e, sr = sf.read(
            os.path.join(self.echo_signal_dir, f"echo_fileid_{aec_idx}.wav")
        )
        u = (
            np.pad(u, (0, max(0, self.max_len - len(u))))
            if len(u) < self.max_len
            else u[: self.max_len]
        )
        e = (
            np.pad(e, (0, max(0, self.max_len - len(e))))
            if len(e) < self.max_len
            else e[: self.max_len]
        )

        if self.aec_roll:
            shift = (
                np.random.randint(0, self.max_len)
                if self.mode == "train"
                else self.aec_rolls[idx]
            )
            u = np.roll(u, shift)
            e = np.roll(e, shift)

        ser = (
            np.random.uniform(self.min_ser, self.max_ser)
            if self.mode == "train"
            else self.sers[idx]
        )
        s_ser_scale = np.sqrt(
            np.abs(e**2).mean() * (10 ** (ser / 10)) / np.abs(s**2).mean()
        )
        s = s * s_ser_scale

        # mix the data
        d = s + e

        return {
            "signals": {
                "u": u[:, None],
                "d": d[:, None],
                "e": e[:, None],
                "s": s[:, None],
            },
            "metadata": {"label": kws_digit, "onehot": self.kws_one_hot_map[kws_digit]},
        }


# from https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
def get_mel_bank(nfilt=30, NFFT=512, sample_rate=16000):
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(
        low_freq_mel, high_freq_mel, nfilt + 2
    )  # Equally spaced in Mel scale
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    return fbank.T


class TCN_KWS(hk.Module):
    def __init__(
        self,
        n_labels,
        n_blocks,
        block_h,
        n_mel,
        kws_window_size,
        kws_hop_size,
        **kwargs,
    ):
        super().__init__()
        self.kws_window_size = kws_window_size
        self.kws_hop_size = kws_hop_size
        self.n_blocks = n_blocks
        self.block_h = block_h

        self.f_bank = get_mel_bank(nfilt=n_mel, NFFT=kws_window_size)
        self.EPS = 1e-12

        self.all_blocks = [
            hk.Sequential(
                [
                    hk.Conv1D(output_channels=block_h // 2, kernel_shape=5, stride=2),
                ]
            )
        ]

        def make_residual(f):
            return lambda x: x + f(x)

        for n in range(self.n_blocks):
            self.all_blocks.append(
                make_residual(
                    hk.Sequential(
                        [
                            hk.Conv1D(output_channels=block_h, kernel_shape=1),
                            hk.LayerNorm(
                                axis=-1, create_scale=True, create_offset=True
                            ),
                            jax.nn.relu,
                            hk.Conv1D(
                                output_channels=block_h,
                                kernel_shape=5,
                                stride=1,
                                rate=2**n,
                                padding="SAME",
                            ),
                            hk.LayerNorm(
                                axis=-1, create_scale=True, create_offset=True
                            ),
                            jax.nn.relu,
                            hk.Conv1D(output_channels=block_h // 2, kernel_shape=1),
                        ]
                    )
                )
            )

        self.all_blocks = hk.Sequential(self.all_blocks)

        self.pred_layer = hk.Sequential(
            [
                hk.Linear(output_size=n_labels),
                jax.nn.softmax,
            ]
        )

    def mel_stft(self, x):
        x = jnp.pad(x, ((0, self.kws_window_size), (0, 0)))
        n_frames = (len(x) - self.kws_window_size) // self.kws_hop_size
        window_idx = jnp.arange(self.kws_window_size)[None, :]
        frame_idx = jnp.arange(n_frames)[:, None]
        window_idxs = window_idx + frame_idx * self.kws_hop_size

        # index the buffer with the map and window
        windowed_x = x[window_idxs] * jnp.hanning(self.kws_window_size)[None, :, None]

        # 0 is T, 1 will be F
        stft_x = jnp.fft.rfft(windowed_x, axis=1)
        return jnp.log10((stft_x.conj() * stft_x).real[..., 0] @ self.f_bank + self.EPS)

    def __call__(self, x_td):
        x_mel = self.mel_stft(x_td)[None]
        tcn_out = self.all_blocks(x_mel)
        pred = self.pred_layer(tcn_out.mean(1))
        return pred[0]

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("KWS")
        parser.add_argument("--n_labels", type=int, default=1)
        parser.add_argument("--n_blocks", type=int, default=1)
        parser.add_argument("--block_h", type=int, default=1)
        parser.add_argument("--n_mel", type=int, default=1)
        parser.add_argument("--kws_window_size", type=int, default=512)
        parser.add_argument("--kws_hop_size", type=int, default=256)
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = [
            "n_labels",
            "n_blocks",
            "block_h",
            "n_mel",
            "kws_window_size",
            "kws_hop_size",
        ]
        return {k: kwargs[k] for k in keys}


def _tcn_kws_fwd(x, kwargs):
    kws = TCN_KWS(**kwargs)
    return kws(x)


@jax.jit
def kws_loss(params, targets, x):
    preds = vec_kws_apply(params, None, x)
    return -np.mean(np.sum(jnp.log(preds) * targets, axis=-1))


def get_accuracy(vec_kws_apply, params, data_loader):
    acc_total = 0
    for batch_idx, data in tqdm.tqdm(enumerate(data_loader)):
        target_class = data["metadata"]["label"]
        predicted_class = np.argmax(
            vec_kws_apply(params, None, data["signals"]["s"]), axis=-1
        )
        acc_total += np.sum(predicted_class == target_class)
    return acc_total / len(data_loader.dataset)


def save_kws_model(kws_params, kws_kwargs, other_kwargs, fname):
    config = {
        "kws_kwargs": kws_kwargs,
        "other_kwargs": other_kwargs,
        "kws_params": kws_params,
    }
    with open(fname, "wb") as file:
        pickle.dump(config, file)


def load_kws_model(loc):
    with open(loc, "rb") as f:
        config = pickle.load(f)
    kws_params = config["kws_params"]
    kws_kwargs = config["kws_kwargs"]
    other_kwargs = config["other_kwargs"]
    kws_model = hk.transform(_tcn_kws_fwd)
    kws_apply = jax.jit(jax.tree_util.Partial(kws_model.apply, kwargs=kws_kwargs))

    return kws_params, kws_apply


"""
python train_kws.py --name 2cmds_tcn --n_labels 2 --block_h 128 --n_blocks 3 --n_mel 40 --kws_window_size 512 --kws_hop_size 256

python train_kws.py --name 10cmds_tcn --n_labels 10 --block_h 128 --n_blocks 3 --n_mel 40 --kws_window_size 512 --kws_hop_size 256

python train_kws.py --name 35cmds_tcn --n_labels 35 --block_h 128 --n_blocks 3 --n_mel 40 --kws_window_size 512 --kws_hop_size 256
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
    parser = TCN_KWS.add_args(parser)
    kwargs = vars(parser.parse_args())

    if kwargs["n_labels"] == 2:
        kws_mode = "2cmds"
    elif kwargs["n_labels"] == 10:
        kws_mode = "10cmds"
    elif kwargs["n_labels"] == 35:
        kws_mode = "35cmds"

    key = jax.random.PRNGKey(0)
    kws_kwargs = TCN_KWS.grab_args(kwargs)
    kws_model = hk.transform(_tcn_kws_fwd)
    data = KWSAECDataset(max_len=64000, mode="train", kws_mode=kws_mode)[0]

    key, subkey = jax.random.split(key)
    params = kws_model.init(key, data["signals"]["s"], kws_kwargs)
    kws_apply = jax.jit(jax.tree_util.Partial(kws_model.apply, kwargs=kws_kwargs))

    vec_kws_apply = jax.vmap(kws_apply, in_axes=(None, None, 0))

    opt_init, opt_update, opt_get_params = optimizers.adam(1e-3)
    opt_state = opt_init(params)

    @jax.jit
    def step_update(opt_state, data):
        l, grads = jax.value_and_grad(kws_loss)(
            opt_get_params(opt_state), data["metadata"]["onehot"], data["signals"]["s"]
        )
        opt_state = opt_update(0, grads, opt_state)
        return l, opt_state

    train_loader = NumpyLoader(
        KWSAECDataset(max_len=64000, mode="train", kws_mode=kws_mode),
        batch_size=128,
        num_workers=5,
        shuffle=True,
    )
    val_loader = NumpyLoader(
        KWSAECDataset(max_len=64000, mode="val", kws_mode=kws_mode), batch_size=128
    )
    test_loader = NumpyLoader(
        KWSAECDataset(max_len=64000, mode="test", kws_mode=kws_mode), batch_size=128
    )

    max_num_epochs = 50
    epoch_train_loss = []
    epoch_val_acc = []

    best_kws_params = params.copy()
    best_val_acc = 0.0

    # Loop over the training epochs
    for epoch in tqdm.tqdm(range(max_num_epochs)):
        train_loss = []
        for data in tqdm.tqdm(train_loader):
            loss, opt_state = step_update(opt_state, data)
            train_loss.append(loss)

        epoch_train_loss.append(np.mean(train_loss))
        epoch_val_acc.append(
            get_accuracy(vec_kws_apply, opt_get_params(opt_state), val_loader)
        )

        if epoch_val_acc[-1] > best_val_acc:
            best_val_acc = epoch_val_acc[-1]
            best_kws_params = opt_get_params(opt_state).copy()

    print(f"Best Val ACC: {best_val_acc}")
    print(epoch_val_acc)

    name = kwargs["name"]
    os.makedirs(kwargs["ckpt_dir"], exist_ok=True)
    save_kws_model(
        best_kws_params,
        kws_kwargs,
        kwargs,
        os.path.join(kwargs["ckpt_dir"], f"{name}.pkl"),
    )
