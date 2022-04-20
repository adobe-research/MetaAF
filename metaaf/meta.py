from tqdm.auto import tqdm
import numpy as np

import jax
from jax import jit
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax.tree_util import Partial, tree_map, tree_leaves
import haiku as hk

from metaaf.core import (
    make_outer_loop,
    make_fit_single,
    make_online_optimizer,
    tree_duplicate,
    tree_split,
)
from metaaf.filter import make_inner_grad
from metaaf.optimizer_gru import (
    _elementwise_gru_fwd,
    ElementWiseGRU,
    init_optimizer,
    make_mapped_optmizer,
)

from metaaf.optimizer_utils import frame_indep_meta_logmse, clip_grads
from metaaf import preprocess_utils as pre_utils
from metaaf import postprocess_utils as post_utils
from metaaf.meta_optimizers import complex_adam


class MetaAFTrainer:
    def __init__(
        self,
        _filter_fwd,
        filter_kwargs,
        filter_loss,
        train_loader,
        val_loader,
        test_loader,
        _optimizer_fwd=_elementwise_gru_fwd,
        optimizer_kwargs=ElementWiseGRU.default_args(),
        init_optimizer=init_optimizer,
        make_mapped_optmizer=make_mapped_optmizer,
        make_get_filter_featues=make_inner_grad,
        _preprocess_fwd=pre_utils._identity_fwd,
        preprocess_kwargs={},
        _postprocess_fwd=post_utils._identity_fwd,
        postprocess_kwargs={},
        meta_train_loss=frame_indep_meta_logmse,
        meta_val_loss=None,
        callbacks=[],
        kwargs={},
    ):
        """Function to initialize a MetaAFTrainer.

        Args:
            _filter_fwd (_type_): Filter forward functin in Haiku formnat
            filter_kwargs (_type_): Dictionary of any filter kwargs
            filter_loss (_type_): The filter loss function
            train_loader (_type_): Pytorch train dataloader that returns dictionary of "signals" to be buffered and "metadata" to be handled by the user
            val_loader (_type_): Pytorch validation dataloader that returns dictionary of "signals" to be buffered and "metadata" to be handled by the user
            test_loader (_type_): Pytorch test dataloader that returns dictionary of "signals" to be buffered and "metadata" to be handled by the user
            _optimizer_fwd (_type_, optional): Optimizer forward function to be applied to the filter. Defaults to _elementwise_gru_fwd.
            optimizer_kwargs (_type_, optional): Optimizer kwarfs. Defaults to ElementWiseGRU.default_args().
            init_optimizer (_type_, optional): Optimizer init function. Defaults to init_optimizer.
            make_mapped_optmizer (_type_, optional): Function to turn optimizer into a mapped optimizer. Defaults to make_mapped_optmizer.
            make_get_filter_featues (_type_, optional): Function to get filter features, typically the grad. Defaults to make_inner_grad.
            _preprocess_fwd (_type_, optional): Preprocessor Haiku style forward. Defaults to pre_utils._identity_fwd.
            preprocess_kwargs (dict, optional): Preprocessors kwargs. Defaults to {}.
            _postprocess_fwd (_type_, optional): Postprocessor Haiku style forward.. Defaults to post_utils._identity_fwd.
            postprocess_kwargs (dict, optional): Postprocessor kwargs. Defaults to {}.
            meta_train_loss (_type_, optional): The meta training loss function. Defaults to frame_indep_meta_logmse.
            meta_val_loss (_type_, optional): The validation function to use for early stopping. Defaults to None.
            callbacks (list, optional): Any optional callbacks. Defaults to [].
            kwargs (dict, optional): Any kwargs that should be logged ans saved via a callback. Defaults to {}.
        """

        # dataset functions
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # optimizer/filter/process forward functions
        self.filter = hk.transform_with_state(_filter_fwd)
        self.optimizer = hk.transform(_optimizer_fwd)
        self.preprocess = hk.transform_with_state(_preprocess_fwd)
        self.postprocess = hk.transform_with_state(_postprocess_fwd)

        # filter kwargs that must be defined here
        self.inner_fixed = {
            "filter_kwargs": filter_kwargs,
        }

        # optimizer/pre/post process kwargs and forward
        self.outer_fixed = {
            # optimizer items
            "optimizer": {
                "optimizer_kwargs": optimizer_kwargs,
                "optimizer": self.optimizer,
            },
            # preprocess items
            "preprocess": {
                "preprocess_kwargs": preprocess_kwargs,
                "preprocess": self.preprocess,
            },
            # postprocess items
            "postprocess": {
                "postprocess_kwargs": postprocess_kwargs,
                "postprocess": self.postprocess,
            },
        }

        # optimizer helper functions
        self.make_mapped_optmizer = make_mapped_optmizer
        self.init_optimizer = init_optimizer
        self.make_get_filter_featues = make_get_filter_featues

        # loss functions
        self.meta_train_loss = meta_train_loss
        self.meta_val_loss = meta_train_loss if meta_val_loss is None else meta_val_loss
        self.filter_loss = filter_loss

        # training hyperparameters
        self.show_progress = True
        self.unroll = None
        self.total_epochs = None
        self.batch_size = None
        self.n_devices = None
        self.meta_opt = None
        self.kwargs = kwargs

        # callbacks
        self.callbacks = callbacks

        # automatically filled in
        self.outer_learnable = None
        self.cur_epoch = 0
        self.cur_batch = 0

        # Compiled/created functions
        self.meta_opt_update = None
        self.meta_opt_get_params = None
        self.outer_loop = None
        self.get_filter_featues = None

        # Compiled/created fast init functions
        self.fast_init_filter = None
        self.fast_init_preprocess = None
        self.fast_init_postprocess = None

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("Trainer")
        parser.add_argument("--unroll", type=int, default=10)
        parser.add_argument("--total_epochs", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--n_devices", type=int, default=1)
        parser.add_argument("--early_stop_patience", type=int, default=np.inf)
        parser.add_argument("--reduce_lr_patience", type=int, default=np.inf)
        parser.add_argument("--val_period", type=int, default=1)
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = [
            "unroll",
            "total_epochs",
            "batch_size",
            "n_devices",
            "early_stop_patience",
            "reduce_lr_patience",
            "val_period",
        ]
        return {k: kwargs[k] for k in keys}

    def init_filter(self, key, batch, batch_init=False):
        if self.fast_init_filter is None:
            # pass a single batch through since we vmap this later
            hop_size = self.inner_fixed["filter_kwargs"]["hop_size"]

            # fast init that takes a new key and data
            @jit
            def fast_init_filter(this_key, this_batch):
                this_hop = tree_map(
                    lambda x: jnp.zeros_like(x[:hop_size]), this_batch["signals"]
                )

                return self.filter.init(
                    this_key,
                    **this_hop,
                    metadata=this_batch["metadata"],
                    init_data=this_batch,
                    **self.inner_fixed["filter_kwargs"],
                )

            self.fast_init_filter = fast_init_filter
            self.fast_batch_init_filter = jax.vmap(fast_init_filter)

        if batch_init:
            return self.fast_batch_init_filter(key, batch)
        return self.fast_init_filter(key, batch)

    def init_preprocess(self, key, batch, batch_init=False):
        if self.fast_init_preprocess is None:
            hop_size = self.inner_fixed["filter_kwargs"]["hop_size"]
            preprocess_dict = self.outer_fixed["preprocess"]

            # fast init that takes a new key and data
            @jit
            def fast_init_preprocess(this_key, this_batch):
                this_hop = tree_map(
                    lambda x: jnp.zeros_like(x[:hop_size]), this_batch["signals"]
                )

                return preprocess_dict["preprocess"].init(
                    this_key,
                    data=this_hop,
                    metadata=this_batch["metadata"],
                    init_data=this_batch,
                    **preprocess_dict["preprocess_kwargs"],
                )

            self.fast_init_preprocess = fast_init_preprocess
            self.fast_batch_init_preprocess = jax.vmap(fast_init_preprocess)

        if batch_init:
            return self.fast_batch_init_preprocess(key, batch)
        return self.fast_init_preprocess(key, batch)

    def init_postprocess(self, key, batch, batch_init=False):
        if self.fast_init_postprocess is None:
            hop_size = self.inner_fixed["filter_kwargs"]["hop_size"]
            postprocess_dict = self.outer_fixed["postprocess"]

            # create a fast jitted init that still takes a new key
            @jit
            def fast_init_postprocess(this_key, this_batch):
                this_hop = tree_map(
                    lambda x: jnp.zeros_like(x[:hop_size]), this_batch["signals"]
                )

                key, subkey = jax.random.split(this_key)
                filter_p, filter_s = self.filter.init(
                    subkey,
                    **this_hop,
                    metadata=this_batch["metadata"],
                    init_data=this_batch,
                    **self.inner_fixed["filter_kwargs"],
                )

                filter_out, _ = self.filter.apply(
                    filter_p,
                    filter_s,
                    None,
                    **this_hop,
                    metadata=this_batch["metadata"],
                    **self.inner_fixed["filter_kwargs"],
                )

                return postprocess_dict["postprocess"].init(
                    key,
                    data=this_hop,
                    metadata=this_batch["metadata"],
                    init_data=this_batch,
                    out=filter_out,
                    **postprocess_dict["postprocess_kwargs"],
                )

            self.fast_init_postprocess = fast_init_postprocess
            self.fast_batch_init_postprocess = jax.vmap(fast_init_postprocess)

        if batch_init:
            return self.fast_batch_init_postprocess(key, batch)
        return self.fast_init_postprocess(key, batch)

    def init_outer_learnable(self, key):
        outer_learnable = {}

        # get batch of size 1
        batch = next(iter(self.train_loader))
        batch = tree_map(lambda x: x[0], batch)

        # initialize a filter
        key, subkey = jax.random.split(key)
        filter_p = self.init_filter(subkey, batch)[0]

        # initialize optimizer
        key, subkey = jax.random.split(key)
        outer_learnable["optimizer_p"] = self.init_optimizer(
            filter_p, batch, self.outer_fixed["optimizer"], subkey
        )

        # pre and post init
        key, subkey = jax.random.split(key)
        outer_learnable["preprocess_p"], _ = self.init_preprocess(subkey, batch)

        key, subkey = jax.random.split(key)
        outer_learnable["postprocess_p"], _ = self.init_postprocess(subkey, batch)

        return outer_learnable

    def init_meta_opt(self, outer_learnable, meta_opt_preprocess):
        self.meta_opt_preprocess = meta_opt_preprocess
        opt_init, update, get_params = self.meta_opt(**self.meta_opt_kwargs)
        self.meta_opt_init = opt_init
        self.meta_opt_update = update
        self.meta_opt_get_params = get_params

        return self.meta_opt_init(outer_learnable)

    def train(
        self,
        unroll=16,
        total_epochs=5,
        batch_size=1,
        n_devices=1,
        early_stop_patience=np.inf,
        reduce_lr_patience=np.inf,
        val_period=1,
        count_first_val=True,
        meta_opt=complex_adam,
        meta_opt_kwargs={"step_size": 1e-4, "b1": 0.99},
        meta_opt_preprocess=Partial(clip_grads, max_norm=10, eps=1e-9),
        key=None,
    ):
        """The actualy train function. Manages the whole metr-training procedure

        Args:
            unroll (int, optional): Integer unroll. Defaults to 16.
            total_epochs (int, optional): Integer num epochs. Defaults to 5.
            batch_size (int, optional): Integer batch size. Defaults to 1.
            n_devices (int, optional): Integer num GPUs. Defaults to 1.
            early_stop_patience (_type_, optional): Patience for early stopping. Defaults to np.inf.
            reduce_lr_patience (_type_, optional): Patience for 1/2 the lr. Defaults to np.inf.
            val_period (int, optional): How often to run validation. Defaults to 1.
            count_first_val (bool, optional): If the very first validation results should be used. Defaults to True.
            meta_opt (_type_, optional): The meta-optimizer. Defaults to optimizers.adam.
            meta_opt_kwargs (dict, optional): Any meta-optimizer kwargs. Defaults to {"step_size": 1e-4, "b1": 0.99}.
            meta_opt_preprocess (_type_, optional): A preprocessor for the meta-grad, like clipping. Defaults to Partial(clip_grads, max_norm=10, eps=1e-9).
            key (_type_, optional): JAX PRNGKey. Defaults to None.

        Returns:
            _type_: The meta-learned parameters and train/val statistics.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        self.unroll = unroll
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.n_devices = n_devices
        self.meta_opt = meta_opt
        self.meta_opt_kwargs = meta_opt_kwargs
        self.early_stop_patience = early_stop_patience
        self.reduce_lr_patience = reduce_lr_patience
        self.val_period = val_period
        self.count_first_val = count_first_val

        # make the outer learnable model
        key, subkey = jax.random.split(key)
        self.outer_learnable = self.init_outer_learnable(subkey)

        # print the parameter count
        n_params = np.sum(
            [
                np.prod(list(l.shape)) if isinstance(l, jnp.ndarray) else 1
                for l in tree_leaves(self.outer_learnable)
            ]
        )
        print(f"Total of - {n_params} - Trainable Parameters")

        # move outer learnable model into the optimizer
        meta_opt_s = self.init_meta_opt(self.outer_learnable, meta_opt_preprocess)

        # init all callbacks
        [
            cb.on_init(
                self.inner_fixed, self.outer_fixed, self.kwargs, self.outer_learnable
            )
            for cb in self.callbacks
        ]

        # train
        self.outer_learnable, aux = self.epoch_loop(meta_opt_s, key)
        return self.outer_learnable, aux

    def epoch_loop(self, meta_opt_s, key):
        pbar = tqdm(
            range(self.total_epochs),
            desc="Epoch Loop",
            leave=True,
            disable=not self.show_progress,
        )

        epoch_train_losses = []
        epoch_val_losses = []
        early_stop_wait = 0
        reduce_lr_wait = 0

        try:
            for epoch in pbar:
                key, subkey = jax.random.split(key)

                # run an epoch in the train loop
                train_loop_losses, meta_opt_s = self.train_loop(
                    epoch=epoch,
                    train_loader=self.train_loader,
                    meta_opt_s=meta_opt_s,
                    key=subkey,
                )
                # save the loop losses
                epoch_train_losses.append(train_loop_losses)

                # update the progress bar
                pbar.set_description(
                    "Epoch Loss:{:.5f}".format(train_loop_losses.mean()), refresh=True
                )

                # collect outer learned parameters
                outer_learnable = self.meta_opt_get_params(meta_opt_s)

                # run validation
                if epoch % self.val_period == 0:
                    key, subkey = jax.random.split(key)
                    val_loop_losses = self.val_loop(outer_learnable, key=subkey)
                    epoch_val_losses.append(val_loop_losses.mean())

                    # only step early stop and lr reduce if this val counts
                    first_val_idx = 0 if self.count_first_val else 1

                    # check for reduce lr option
                    reduce_lr_wait = self.get_val_increases(
                        reduce_lr_wait, epoch_val_losses[first_val_idx:]
                    )

                    # learning rate divide by 2 patience
                    if reduce_lr_wait >= self.reduce_lr_patience:
                        self.meta_opt_kwargs["step_size"] = (
                            self.meta_opt_kwargs["step_size"] / 2
                        )
                        self.meta_opt_update = self.meta_opt(**self.meta_opt_kwargs)[1]
                        reduce_lr_wait = 0

                    early_stop_wait = self.get_val_increases(
                        early_stop_wait, epoch_val_losses[first_val_idx:]
                    )

                    # check the early stopping patience
                    if early_stop_wait >= self.early_stop_patience:
                        # log final epoch end call back
                        [
                            cb.on_train_epoch_end(
                                train_loop_losses,
                                val_loop_losses,
                                outer_learnable,
                                self.cur_epoch,
                            )
                            for cb in self.callbacks
                        ]
                        break

                # train epoch end all callbacks
                [
                    cb.on_train_epoch_end(
                        train_loop_losses,
                        val_loop_losses,
                        outer_learnable,
                        self.cur_epoch,
                    )
                    for cb in self.callbacks
                ]

                # step the epochs
                self.cur_epoch += 1
        except KeyboardInterrupt:
            print("Training Interupted, trying to shutdown gracefully.")

        # collect outer learned parameters
        outer_learnable = self.meta_opt_get_params(meta_opt_s)
        aux = (np.array(epoch_train_losses), np.array(epoch_val_losses))

        # train end all callbacks
        [
            cb.on_train_end(
                epoch_train_losses, epoch_val_losses, outer_learnable, self.cur_epoch
            )
            for cb in self.callbacks
        ]

        return outer_learnable, aux

    def train_loop(self, epoch, train_loader, meta_opt_s, key):
        # duplicate meta_opt_s across devices
        meta_opt_s_stack = tree_duplicate(meta_opt_s, self.n_devices)
        train_loop_losses = []

        pbar = tqdm(
            train_loader, desc="Batch Loop", leave=False, disable=not self.show_progress
        )

        for batch in pbar:
            # move batch data to JAX
            batch = tree_map(lambda x: jnp.array(x), batch)
            batch_size = next(iter(batch["signals"].items()))[1].shape[0]

            # initialize filter and processing
            key, *subkeys = jax.random.split(key, 1 + batch_size)
            filter_p, filter_s = self.init_filter(
                jnp.array(subkeys), batch, batch_init=True
            )

            key, *subkeys = jax.random.split(key, 1 + batch_size)
            preprocess_s = self.init_preprocess(
                jnp.array(subkeys), batch, batch_init=True
            )[
                1
            ]  # get state not params

            key, *subkeys = jax.random.split(key, 1 + batch_size)
            postprocess_s = self.init_postprocess(
                jnp.array(subkeys), batch, batch_init=True
            )[
                1
            ]  # get state not params

            # reshape across devices for pmap
            batch = tree_split(batch, self.n_devices)
            filter_p = tree_split(filter_p, self.n_devices)
            filter_s = tree_split(filter_s, self.n_devices)
            preprocess_s = tree_split(preprocess_s, self.n_devices)
            postprocess_s = tree_split(postprocess_s, self.n_devices)

            # run the train step
            key, *subkeys = jax.random.split(key, 1 + self.n_devices)
            train_step_losses, aux = self.train_step(
                meta_opt_s=meta_opt_s_stack,
                filter_p=filter_p,
                filter_s=filter_s,
                preprocess_s=preprocess_s,
                postprocess_s=postprocess_s,
                batch=batch,
                key=jnp.array(subkeys),
            )
            _, _, meta_opt_s_stack = aux

            train_loop_losses.append(train_step_losses.mean())

            pbar.set_description(
                "Batch Loss:{:.5f}".format(train_step_losses.mean()), refresh=True
            )

            self.cur_batch += 1

            # train step end all callbacks
            [
                cb.on_train_batch_end(
                    train_step_losses, aux, batch, self.cur_batch, self.cur_epoch
                )
                for cb in self.callbacks
            ]

        # retrieve a single meta_opt_s from the stack
        meta_opt_s = jax.tree_util.tree_map(lambda x: x[0], meta_opt_s_stack)

        return np.array(train_loop_losses), meta_opt_s

    def train_step(
        self,
        meta_opt_s,
        filter_p,
        filter_s,
        preprocess_s,
        postprocess_s,
        batch,
        key,
    ):
        if self.get_filter_featues is None:
            self.get_filter_featues = self.make_get_filter_featues(
                self.filter, self.inner_fixed, self.filter_loss
            )

        if self.outer_loop is None:
            outer_loop = make_outer_loop(
                make_mapped_optmizer=self.make_mapped_optmizer,
                outer_fixed=self.outer_fixed,
                meta_loss=self.meta_train_loss,
                meta_opt_update=self.meta_opt_update,
                meta_opt_get_params=self.meta_opt_get_params,
                meta_opt_preprocess=self.meta_opt_preprocess,
                get_filter_featues=self.get_filter_featues,
                unroll=self.unroll,
                hop_size=self.inner_fixed["filter_kwargs"]["hop_size"],
            )

            self.outer_loop = jax.pmap(outer_loop, axis_name="devices")

        train_step_losses, aux = self.outer_loop(
            meta_opt_s=meta_opt_s,
            filter_p=filter_p,
            filter_s=filter_s,
            preprocess_s=preprocess_s,
            postprocess_s=postprocess_s,
            batch=batch,
            key=key,
        )
        return train_step_losses, aux

    def get_val_increases(self, n_val_increases, epoch_val_losses):
        # check if val loss improved
        if len(epoch_val_losses) > 1 and epoch_val_losses[-1] > min(
            epoch_val_losses[:-1]
        ):
            # if current loss is worse than best of all previous increments
            n_val_increases += 1
        else:
            # if current loss was better reset the patience
            n_val_increases = 0
        return n_val_increases

    def val_loop(self, outer_learnable=None, key=None):
        """Can be called to run validation.

        Args:
            outer_learnable (_type_, optional): Outer learnable parameters returned by train or stored internally. Defaults to None.
            key (_type_, optional): JAX PRNGKey. Defaults to None.

        Returns:
            _type_: Returns the mean loss but also calls the callbacks.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if outer_learnable is None:
            outer_learnable = self.outer_learnable

        if self.get_filter_featues is None:
            self.get_filter_featues = self.make_get_filter_featues(
                self.filter, self.inner_fixed, self.filter_loss
            )
        # precompile the fit single batch functin
        fit_infer = self.make_fit_infer(outer_learnable=outer_learnable)

        loop_loss = []
        for (batch_idx, batch) in enumerate(self.val_loader):
            key, subkey = jax.random.split(key)
            out, aux = self.infer(
                batch,
                outer_learnable=outer_learnable,
                fit_infer=fit_infer,
                key=subkey,
            )

            # call backs on end of a val batch
            [
                cb.on_val_batch_end(out, aux, batch, batch_idx, self.cur_epoch)
                for cb in self.callbacks
            ]

            loss = aux[0]
            loop_loss.append(loss)

        loop_loss = np.array(loop_loss)

        # call backs at end of val
        [cb.on_val_end(loop_loss, self.cur_epoch) for cb in self.callbacks]

        return loop_loss

    def test_loop(self, outer_learnable=None, key=None):
        """Can be called to run testing.

        Args:
            outer_learnable (_type_, optional): Outer learnable parameters returned by train or stored internally. Defaults to None.
            key (_type_, optional): JAX PRNGKey. Defaults to None.

        Returns:
            _type_: Returns the mean loss but also calls the callbacks.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if outer_learnable is None:
            outer_learnable = self.outer_learnable

        if self.get_filter_featues is None:
            self.get_filter_featues = self.make_get_filter_featues(
                self.filter, self.inner_fixed, self.filter_loss
            )

        # precompile the fit single batch functin
        fit_infer = self.make_fit_infer(outer_learnable=outer_learnable)

        loop_loss = []
        for (batch_idx, batch) in enumerate(self.test_loader):
            key, subkey = jax.random.split(key)
            out, aux = self.infer(
                batch,
                outer_learnable=outer_learnable,
                fit_infer=fit_infer,
                key=subkey,
            )
            # call backs on end of a test batch
            [
                cb.on_test_batch_end(out, aux, batch, batch_idx, self.cur_epoch)
                for cb in self.callbacks
            ]

            loss = aux[0]
            loop_loss.append(loss)

        loop_loss = np.array(loop_loss)

        # call backs at end of test
        [cb.on_test_end(loop_loss, self.cur_epoch) for cb in self.callbacks]

        return loop_loss

    def make_fit_infer(self, outer_learnable=None):
        """Makes the fit function that can be used for one-off or custom testing.

        Args:
            outer_learnable (_type_, optional): Outer learnable parameters returned by train or stored internally.. Defaults to None.

        Returns:
            _type_: Fit function.
        """
        if outer_learnable is None:
            outer_learnable = self.outer_learnable

        get_filter_featues = self.make_get_filter_featues(
            self.filter, self.inner_fixed, self.filter_loss
        )
        return make_fit_single(
            outer_learnable=outer_learnable,
            outer_fixed=self.outer_fixed,
            meta_loss=self.meta_val_loss,
            make_mapped_optmizer=self.make_mapped_optmizer,
            get_filter_featues=get_filter_featues,
            hop_size=self.inner_fixed["filter_kwargs"]["hop_size"],
        )

    def infer(
        self,
        batch,
        outer_learnable=None,
        fit_infer=None,
        filter_p=None,
        filter_s=None,
        preprocess_s=None,
        postprocess_s=None,
        key=None,
    ):
        """Function to trun inference

        Args:
            batch (_type_): Data to process
            outer_learnable (_type_, optional): Outer learnable parameters returned by train or stored internally. Defaults to None.
            fit_infer (_type_, optional): Optionally jitted infer. Defaults to None.
            filter_p (_type_, optional): Filter parameters. Defaults to None.
            filter_s (_type_, optional): Filter state. Defaults to None.
            preprocess_s (_type_, optional): Preprocess state. Defaults to None.
            postprocess_s (_type_, optional): Postprocess state. Defaults to None.
            key (_type_, optional): JAX PRNGKey. Defaults to None.

        Returns:
            _type_: Returns tuple of outputs and then any auxilliary data
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if outer_learnable is None:
            outer_learnable = self.outer_learnable

        if self.get_filter_featues is None:
            self.get_filter_featues = self.make_get_filter_featues(
                self.filter, self.inner_fixed, self.filter_loss
            )
        if fit_infer is None:
            fit_infer = self.make_fit_infer(outer_learnable=outer_learnable)

        if filter_p is None and filter_s is None:
            batch_size = next(iter(batch["signals"].items()))[1].shape[0]
            key, *subkeys = jax.random.split(key, 1 + batch_size)
            filter_p, filter_s = self.init_filter(
                jnp.array(subkeys), batch, batch_init=True
            )

        if preprocess_s is None:
            batch_size = next(iter(batch["signals"].items()))[1].shape[0]
            key, *subkeys = jax.random.split(key, 1 + batch_size)
            preprocess_s = self.init_preprocess(
                jnp.array(subkeys), batch, batch_init=True
            )[1]

        if postprocess_s is None:
            batch_size = next(iter(batch["signals"].items()))[1].shape[0]
            key, *subkeys = jax.random.split(key, 1 + batch_size)
            postprocess_s = self.init_postprocess(
                jnp.array(subkeys), batch, batch_init=True
            )[1]

        out, aux = fit_infer(
            filter_s, filter_p, preprocess_s, postprocess_s, batch, key
        )

        return out, aux

    def make_online_infer(self, outer_learnable=None, key=None):
        """Makes functions to process data in a streaming fashion.

        Args:
            outer_learnable (_type_, optional): Outer learnable parameters returned by train or stored internally. Defaults to None.
            key (_type_, optional): JAX PRNGKey. Defaults to None.

        Returns:
            _type_: Returns an online_step and an initial_state.
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        if outer_learnable is None:
            outer_learnable = self.outer_learnable

        if self.get_filter_featues is None:
            self.get_filter_featues = self.make_get_filter_featues(
                self.filter, self.inner_fixed, self.filter_loss
            )

        online_step, online_state = make_online_optimizer(
            outer_learnable=outer_learnable,
            outer_fixed=self.outer_fixed,
            make_mapped_optmizer=self.make_mapped_optmizer,
            get_filter_featues=self.get_filter_featues,
        )

        # get batch of size 1
        batch = next(iter(self.train_loader))
        batch = tree_map(lambda x: x[0], batch)

        key, subkey = jax.random.split(key)
        filter_p, filter_s = self.init_filter(subkey, batch)

        key, subkey = jax.random.split(key)
        preprocess_s = self.init_preprocess(subkey, batch)[1]

        key, subkey = jax.random.split(key)
        postprocess_s = self.init_postprocess(subkey, batch)[1]

        initial_state = (
            filter_s,
            online_state(filter_p),
            preprocess_s,
            postprocess_s,
            0,
        )
        return online_step, initial_state
