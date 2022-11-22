from abc import ABCMeta
import os
import json
import shutil
import pickle
import datetime
import numpy as np
import wandb
import soundfile as sf


class CallbackMetaClass(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def on_init(self, inner_fixed, outer_fixed, kwargs, outer_learnable):
        pass

    def on_train_batch_end(
        self, train_step_losses, aux, data_batch, cur_batch, cur_epoch
    ):
        pass

    def on_train_epoch_end(
        self, train_loop_losses, val_loop_losses, outer_learnable, cur_epoch
    ):
        pass

    def on_train_end(
        self, epoch_train_losses, epoch_val_losses, outer_learnable, cur_epoch
    ):
        pass

    def on_val_batch_end(self, out, aux, data_batch, cur_batch, cur_epoch):
        pass

    def on_val_end(self, val_losses, cur_epoch):
        pass

    def on_test_batch_end(self, out, aux, data_batch, cur_batch, cur_epoch):
        pass

    def on_test_end(self, val_losses, cur_epoch):
        pass


class DebugCallBack(CallbackMetaClass):
    def __init__(self):
        super().__init__()

    def on_init(self, inner_fixed, outer_fixed, kwargs, outer_learnable):
        print("On Init")

    def on_train_batch_end(
        self, train_step_losses, aux, data_batch, cur_batch, cur_epoch
    ):
        print("On train batch end")

    def on_train_epoch_end(
        self, train_loop_losses, val_loop_losses, outer_learnable, cur_epoch
    ):
        print("On train epoch end")

    def on_train_end(
        self, epoch_train_losses, epoch_val_losses, outer_learnable, cur_epoch
    ):
        print("On train end")

    def on_val_batch_end(self, out, aux, data_batch, cur_batch, cur_epoch):
        print("On val batch end")

    def on_val_end(self, val_losses, cur_epoch):
        print("On val end")

    def on_test_batch_end(self, out, aux, data_batch, cur_batch, cur_epoch):
        print("On test batch end")

    def on_test_end(self, val_losses, cur_epoch):
        print("On test end")


class CheckpointCallback(CallbackMetaClass):
    def __init__(self, name, ckpt_base_dir):
        super().__init__()
        # setup the checkpoint dir
        self.start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.ckpt_dir = os.path.join(ckpt_base_dir, name, self.start_time)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        # setup to identify the best model
        self.best_val_loss = np.inf
        self.best_epoch = 0

    def on_init(self, inner_fixed, outer_fixed, kwargs, outer_learnable):
        all_kwargs = {
            **inner_fixed["filter_kwargs"],
            **outer_fixed["optimizer"]["optimizer_kwargs"],
            **outer_fixed["preprocess"]["preprocess_kwargs"],
            **outer_fixed["postprocess"]["postprocess_kwargs"],
            **kwargs,
        }

        with open(os.path.join(self.ckpt_dir, "all_kwargs.json"), "w") as file:
            json.dump(all_kwargs, file)

    def on_train_epoch_end(
        self, train_loop_losses, val_loop_losses, outer_learnable, cur_epoch
    ):
        # save the model
        fname = os.path.join(self.ckpt_dir, f"epoch_{cur_epoch}.pkl")
        with open(fname, "wb") as file:
            pickle.dump(outer_learnable, file)

        # check if this was the best model
        cur_val_loss = val_loop_losses.mean()
        if cur_val_loss < self.best_val_loss:
            self.best_val_loss = cur_val_loss
            self.best_epoch = cur_epoch

    def on_train_end(
        self, epoch_train_losses, epoch_val_losses, outer_learnable, cur_epoch
    ):
        best_ckpt_dir = os.path.join(self.ckpt_dir, f"epoch_{self.best_epoch}.pkl")
        best_ckpt_name = os.path.join(
            self.ckpt_dir, f"best_ckpt_epoch_{self.best_epoch}.pkl"
        )
        shutil.copy(best_ckpt_dir, best_ckpt_name)


class WandBCallback(CallbackMetaClass):
    def __init__(self, project, name, entity, log_pd=20):
        super().__init__()
        self.log_pd = log_pd
        wandb.init(project=project, name=name, entity=entity)

    def on_init(self, inner_fixed, outer_fixed, kwargs, outer_learnable):
        all_kwargs = {
            **inner_fixed["filter_kwargs"],
            **outer_fixed["optimizer"]["optimizer_kwargs"],
            **outer_fixed["preprocess"]["preprocess_kwargs"],
            **outer_fixed["postprocess"]["postprocess_kwargs"],
            **kwargs,
        }
        wandb.config.update(all_kwargs)

    def on_train_batch_end(
        self, train_step_losses, aux, data_batch, cur_batch, cur_epoch
    ):
        if cur_batch % self.log_pd == 0:
            wandb.log(
                {
                    "train_loss": np.nanmean(np.array(train_step_losses)),
                    "train_num_nan": np.isnan(np.array(train_step_losses)).sum(),
                    "epoch": cur_epoch,
                    "batch": cur_batch,
                }
            )

    def on_train_epoch_end(
        self, train_loop_losses, val_loop_losses, outer_learnable, cur_epoch
    ):
        wandb.log(
            {
                "val_loss": np.nanmean(np.array(val_loop_losses)),
                "val_num_nan": np.isnan(np.array(val_loop_losses)).sum(),
                "epoch": cur_epoch,
            }
        )


class AudioLoggerCallback(CallbackMetaClass):
    def __init__(self, name, outputs_base_dir, num_to_log=5, fs=16000):
        super().__init__()
        # setup the output dir
        self.start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.output_dir = os.path.join(outputs_base_dir, name, self.start_time)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.num_to_log = num_to_log
        self.num_logged = 0
        self.fs = fs

    def on_val_batch_end(self, out, aux, data_batch, cur_batch, cur_epoch):
        # create a folder for this epoch
        epoch_dir = os.path.join(self.output_dir, f"epoch_{cur_epoch}")
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        # only log num_to_log at each epoch
        batch_idx = 0
        while self.num_logged < self.num_to_log and batch_idx < len(out):
            base_name = os.path.join(epoch_dir, f"{self.num_logged}")
            sf.write(f"{base_name}_out.wav", np.array(out[batch_idx, :, 0]), self.fs)

            for (k, v) in data_batch["signals"].items():
                sf.write(f"{base_name}_{k}.wav", np.array(v[batch_idx, :, 0]), self.fs)

            batch_idx += 1
            self.num_logged += 1

    def on_val_end(self, val_losses, cur_epoch):
        self.num_logged = 0
