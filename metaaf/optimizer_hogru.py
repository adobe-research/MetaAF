import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import haiku as hk
import numpy as np
from metaaf.complex_gru import CGRU, add_batch
from metaaf.complex_utils import complex_variance_scaling, complex_zeros, complex_relu
from metaaf.complex_groupnorm import CGN


def make_deep_initial_state(params, **kwargs):
    c_size = kwargs["c_size"]
    h_size = kwargs["h_size"]
    n_layers = kwargs["n_layers"]

    def single_layer_initial_state():
        state = jnp.zeros([h_size], dtype=np.dtype("complex64"))
        state = add_batch(state, c_size)
        return state

    return tuple(single_layer_initial_state() for _ in range(n_layers))


class HOElementWiseGRU(hk.Module):
    def __init__(
        self,
        h_size,
        n_layers,
        lam,
        freq_size,
        c_size,
        group_mode="block",
        group_size=5,
        input_transform="log1p",
        name="HOElementWiseGRU",
        **kwargs
    ):

        super().__init__(name=name)
        self.h_size = h_size
        self.n_layers = n_layers
        self.lam = lam
        self.freq_size = freq_size
        self.c_size = c_size

        assert input_transform in ["raw", "log1p", "logclip", "bct"]
        assert group_mode in ["diag", "block", "banded"]

        self.input_transform = input_transform

        self.group_size = group_size
        self.group_mode = group_mode

        # padding 0s for block group
        if self.group_mode == "block":
            self.freq_size_padded = self.c_size * self.group_size
            padding_size = self.freq_size_padded - self.freq_size

            self.rp = padding_size // 2
            self.lp = padding_size - self.rp

        if self.group_mode != "banded":
            self.in_lin = hk.Sequential(
                [
                    hk.Linear(
                        output_size=self.h_size,
                        w_init=complex_variance_scaling,
                        b_init=complex_zeros,
                    ),
                ]
            )

            self.out_lin = hk.Sequential(
                [
                    hk.Linear(
                        output_size=self.group_size,
                        w_init=complex_variance_scaling,
                        b_init=complex_zeros,
                    ),
                ]
            )

        else:
            in_coupling_conv = [
                hk.Conv1D(
                    output_channels=self.h_size,
                    kernel_shape=self.group_size,
                    stride=self.group_size // 2,
                    w_init=complex_variance_scaling,
                    b_init=complex_zeros,
                ),
                CGN(groups=self.h_size // 8),
                complex_relu,
            ]
            self.in_coupling_conv = hk.Sequential(in_coupling_conv)

            out_coupling_conv = [
                hk.Conv1DTranspose(
                    output_channels=1,
                    kernel_shape=self.group_size,
                    stride=self.group_size // 2,
                    output_shape=self.freq_size,
                    w_init=complex_variance_scaling,
                    b_init=complex_zeros,
                ),
            ]
            self.out_coupling_conv = hk.Sequential(out_coupling_conv)

        self.rnn_stack = hk.DeepRNN(
            [CGRU(hidden_size=self.h_size) for _ in range(n_layers)]
        )

    def preprocess_flatten(self, x, extra_inputs):
        # stack the gradients as well as any extra inputs
        input_stack = jnp.stack((x, *extra_inputs), axis=-1)

        # feature extraction options
        if self.input_transform == "log1p":
            mag = jnp.log1p(jnp.abs(input_stack))
            phase = jnp.exp(1.0j * jnp.angle(input_stack))
            input_stack = mag * phase
        elif self.input_transform == "logclip":
            p = 25
            mag = jax.lax.clamp(jnp.exp(-p), jnp.abs(input_stack), jnp.exp(p))
            mag = jnp.log(mag + 1e-8) / p + 1
            phase = jnp.exp(1.0j * jnp.angle(input_stack))
            input_stack = mag * phase
        elif self.input_transform == "bct":
            l = 0.2
            mag = ((jnp.abs(input_stack) + 1e-10) ** l - 1) / l
            phase = jnp.exp(1.0j * jnp.angle(input_stack))
            input_stack = mag * phase

        if self.group_mode == "block":
            if self.lp != 0:
                # pad to be divisible by the group size
                input_stack_pad = jnp.pad(input_stack, (self.rp, self.lp))[
                    self.rp : -self.lp, :, self.rp : -self.lp, self.rp : -self.lp
                ]
            else:
                input_stack_pad = input_stack

            input_stack_group = input_stack_pad.reshape(
                1, self.c_size, 1, 5 * self.group_size
            )
            input_stack_flat = input_stack_group.reshape((-1, 5 * self.group_size))

        elif self.group_mode == "banded":
            input_stack_flat = self.in_coupling_conv(
                input_stack.reshape(-1, input_stack.shape[-1])
            )

        else:
            input_stack_flat = input_stack.reshape((-1, input_stack.shape[-1]))

        if self.group_mode == "banded":
            return input_stack_flat
        else:
            return self.in_lin(input_stack_flat)

    def postprocess_reshape(self, rnn_out, raw_input):

        if self.group_mode == "block":
            reshaped_out = rnn_out.reshape((*raw_input.shape, rnn_out.shape[-1]))
            reshaped_out = self.out_lin(reshaped_out)
            reshaped_out = reshaped_out.reshape(1, self.freq_size_padded, 1, 1)[
                :, : self.freq_size, :, :
            ]

        elif self.group_mode == "banded":
            reshaped_out = self.out_coupling_conv(rnn_out)
            reshaped_out = reshaped_out.reshape((1, self.freq_size, 1, 1))

        else:
            reshaped_out = rnn_out.reshape((*raw_input.shape, rnn_out.shape[-1]))
            reshaped_out = self.out_lin(reshaped_out)

        out = reshaped_out[..., 0]
        return -self.lam * out

    def __call__(self, x, h, extra_inputs):
        # do feature extraction like log1p and input coupling
        rnn_in = self.preprocess_flatten(x, extra_inputs)

        # take the RNN step
        rnn_out, h = self.rnn_stack(rnn_in, h)

        # reassemble to right shape and do any output coupling
        out = self.postprocess_reshape(rnn_out, x[:, : self.c_size, :])

        return out, h

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("Optimizer")
        parser.add_argument("--h_size", type=int, default=32)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--group_mode", type=str, default="manual")
        parser.add_argument("--group_size", type=int, default=4)
        parser.add_argument("--input_transform", type=str, default="log1p")
        parser.add_argument("--lam", type=float, default=0.01)
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = [
            "h_size",
            "n_layers",
            "lam",
            "freq_size",
            "c_size",
            "group_mode",
            "group_size",
            "input_transform",
        ]
        return {k: kwargs[k] for k in keys}

    @staticmethod
    def default_args():
        return {
            "h_size": 32,
            "n_layers": 2,
            "lam": 0.01,
            "freq_size": 2049,
            "c_size": 410,
            "group_mode": "block",
            "group_size": 5,
            "input_transform": "log1p",
        }


def _elementwise_hogru_fwd(x, h, *extra_inputs, **kwargs):
    optimizer = HOElementWiseGRU(**kwargs)
    return optimizer(x, h, extra_inputs)


class Identity(hk.Module):
    def __init__(self, name="Identity", **kwargs):

        super().__init__(name=name)

    def __call__(self, x, h, extra_inputs):

        return x, h

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("Optimizer")
        parser.add_argument("--h_size", type=int, default=32)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--group_mode", type=str, default="block")
        parser.add_argument("--group_size", type=int, default=4)
        parser.add_argument("--input_transform", type=str, default="log1p")
        parser.add_argument("--lam", type=float, default=0.01)
        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = [
            "h_size",
            "n_layers",
            "lam",
            "freq_size",
            "c_size",
            "group_mode",
            "group_size",
            "input_transform",
        ]
        return {k: kwargs[k] for k in keys}

    @staticmethod
    def default_args():
        return {
            "h_size": 32,
            "n_layers": 2,
            "lam": 0.01,
            "freq_size": 2049,
            "c_size": 410,
            "group_mode": "block",
            "group_size": 5,
            "input_transform": "log1p",
        }


def _identity_fwd(x, h, *extra_inputs, **kwargs):
    optimizer = Identity(**kwargs)
    return optimizer(x, h, extra_inputs)


def init_optimizer(optimizee_p, batch_data, optimizer_dict, key):
    single_p = jax.tree_util.tree_leaves(optimizee_p)[0]

    # use that parameter to make an optimizer state
    h = make_deep_initial_state(single_p, **optimizer_dict["optimizer_kwargs"])

    # now init the optimizer
    optimizer_p = optimizer_dict["optimizer"].init(
        key, single_p, h, **optimizer_dict["optimizer_kwargs"]
    )

    return optimizer_p


def init_optimizer_all_data(optimizee_p, batch_data, optimizer_dict, key):
    single_p = jax.tree_util.tree_leaves(optimizee_p)[0]

    # use that parameter to make an optimizer state
    h = make_deep_initial_state(single_p, **optimizer_dict["optimizer_kwargs"])

    print("input_stack", single_p.shape)

    # now init the optimizer
    optimizer_p = optimizer_dict["optimizer"].init(
        key,
        single_p,
        h,
        single_p,
        single_p,
        single_p,
        single_p,
        **optimizer_dict["optimizer_kwargs"],
    )

    return optimizer_p


def init_optimizer_identity(optimizee_p, batch_data, optimizer_dict, key):
    single_p = jax.tree_util.tree_leaves(optimizee_p)[0]

    # use that parameter to make an optimizer state
    h = None

    # now init the optimizer
    optimizer_p = optimizer_dict["optimizer"].init(
        key, single_p, h, **optimizer_dict["optimizer_kwargs"]
    )

    return optimizer_p


@optimizers.optimizer
def make_mapped_optmizer(optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs):
    def init(optimizee_p):
        state = make_deep_initial_state(optimizee_p, **optimizer_kwargs)
        return (optimizee_p, state)

    def update(i, features, jax_state):
        optimizee_p, state = jax_state
        update, state = optimizer.apply(
            optimizer_p,
            None,
            jnp.conj(features.optimizee_features),
            state,
            **optimizer_kwargs,
        )

        return (optimizee_p + update, state)

    def get_params(jax_state):
        # state was (parameters, state)
        return jax_state[0]

    return init, update, get_params


@optimizers.optimizer
def make_mapped_optmizer_all_data(
    optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs
):
    def init(optimizee_p):
        state = make_deep_initial_state(optimizee_p, **optimizer_kwargs)
        return (optimizee_p, state)

    def update(i, features, jax_state):
        optimizee_p, state = jax_state

        u = features.cur_outputs["u"]
        d = jnp.broadcast_to(features.cur_outputs["d"], u.shape)
        y = jnp.broadcast_to(features.cur_outputs["y"], u.shape)
        e = jnp.broadcast_to(features.cur_outputs["e"], u.shape)
        grad = features.cur_outputs["grad"]

        update, state = optimizer.apply(
            optimizer_p,
            None,
            grad,
            state,
            u,
            d,
            e,
            y,
            **optimizer_kwargs,
        )

        return (optimizee_p + update, state)

    def get_params(jax_state):
        # state was (parameters, state)
        return jax_state[0]

    return init, update, get_params


@optimizers.optimizer
def make_mapped_optmizer_all_data_iwaenc(
    optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs
):
    def init(optimizee_p):
        state = make_deep_initial_state(optimizee_p, **optimizer_kwargs)
        return (optimizee_p, state)

    def update(i, features, jax_state):
        optimizee_p, state = jax_state

        u = features.cur_outputs["u"]
        d = jnp.broadcast_to(features.cur_outputs["d"], u.shape)
        y = jnp.broadcast_to(features.cur_outputs["y"], u.shape)
        e = jnp.broadcast_to(features.cur_outputs["e"], u.shape)

        update, state = optimizer.apply(
            optimizer_p,
            None,
            jnp.conj(features.filter_features),
            state,
            u,
            d,
            y,
            e,
            **optimizer_kwargs,
        )

        return (optimizee_p + update, state)

    def get_params(jax_state):
        # state was (parameters, state)
        return jax_state[0]

    return init, update, get_params


@optimizers.optimizer
def make_mapped_optmizer_identity(
    optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs
):
    def init(optimizee_p):
        state = None
        return (optimizee_p, state)

    def update(i, features, jax_state):
        optimizee_p, state = jax_state
        update, state = optimizer.apply(
            optimizer_p,
            None,
            jnp.conj(features.filter_features),
            state,
            **optimizer_kwargs,
        )

        return (optimizee_p + update, state)

    def get_params(jax_state):
        # state was (parameters, state)
        return jax_state[0]

    return init, update, get_params
