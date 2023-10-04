import numpy as np
import soundfile as sf
import os
import json

from torch.utils.data import Dataset

from zoo.__config__ import AEC_DATA_DIR


class MSFTAECRealDataset(Dataset):
    def __init__(
        self,
        base_dir=AEC_DATA_DIR,
        mode="train",
        train_dict="id2ID_dict_real_1_2.json",
        val_dict="id2ID_dict_test_set_1.json",
        test_dict="id2ID_dict_blind_test_set_1_2.json",
    ):
        synthetic_dir = os.path.join(base_dir, "datasets/synthetic/")

        self.echo_signal_dir = os.path.join(synthetic_dir, "echo_signal")
        self.farend_speech_dir = os.path.join(synthetic_dir, "farend_speech")
        self.nearend_mic_dir = os.path.join(synthetic_dir, "nearend_mic_signal")
        self.nearend_speech_dir = os.path.join(synthetic_dir, "nearend_speech")

        self.mode = mode

        if self.mode == "test_real":
            with open(test_dict) as json_file:
                self.id2ID_dict = json.load(json_file)
            self.offset = 0
        elif self.mode == "test_synthetic":
            self.offset = 0
        elif self.mode == "val_real":
            with open(val_dict) as json_file:
                self.id2ID_dict = json.load(json_file)
            self.offset = 0
        elif self.mode == "val_synthetic":
            self.offset = 500
        else:
            with open(train_dict) as json_file:
                self.id2ID_dict = json.load(json_file)
            self.offset = 0

    def __len__(self):
        if self.mode == "test_real":
            return 800
        elif self.mode == "test_synthetic":
            return 500
        elif self.mode == "val_real":
            return 196
        elif self.mode == "val_synthetic":
            return 500
        else:
            return 7282 + 6341

    def load_from_idx(self, idx):
        if self.mode == "val_synthetic" or self.mode == "test_synthetic":
            idx = idx + self.offset
            d, sr = sf.read(
                os.path.join(self.nearend_mic_dir, f"nearend_mic_fileid_{idx}.wav")
            )

            u, sr = sf.read(
                os.path.join(self.farend_speech_dir, f"farend_speech_fileid_{idx}.wav")
            )

            e, sr = sf.read(
                os.path.join(self.echo_signal_dir, f"echo_fileid_{idx}.wav")
            )
            s, sr = sf.read(
                os.path.join(
                    self.nearend_speech_dir, f"nearend_speech_fileid_{idx}.wav"
                )
            )

        else:
            # idx = idx + self.offset
            d, sr = sf.read(
                self.id2ID_dict[str(idx)]["mic"].replace(
                    "/mnt/data/AEC/new_AEC-Challenge", AEC_DATA_DIR
                )
            )

            u, sr = sf.read(
                self.id2ID_dict[str(idx)]["lpb"].replace(
                    "/mnt/data/AEC/new_AEC-Challenge", AEC_DATA_DIR
                )
            )

            e = d[:160000]
            s = d[:160000]  # np.zeros(160000)

            d = d[:160000]
            u = u[:160000]

        u = np.pad(u, (0, max(0, 160000 - len(u))))
        d = np.pad(d, (0, max(0, 160000 - len(d))))
        e = np.pad(e, (0, max(0, 160000 - len(e))))
        s = np.pad(s, (0, max(0, 160000 - len(s))))

        data_dict = {"d": d[:, None], "u": u[:, None], "e": e[:, None], "s": s[:, None]}
        return {"signals": data_dict, "metadata": {}}

    def __getitem__(self, idx):
        data_dict = self.load_from_idx(idx)
        return data_dict
