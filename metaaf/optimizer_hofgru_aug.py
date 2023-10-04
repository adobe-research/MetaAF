import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import haiku as hk
from metaaf.complex_gru import CGRU, add_batch
from metaaf.complex_utils import (
    complex_zeros,
    complex_relu,
    complex_xavier,
)
from metaaf.complex_norm import CGN


class HO_FGRU_AUG(hk.Module):
    def __init__(
        self,
        h_size,
        n_layers,
        lam_1,
        outsize,
        group_size,
        group_hop,
        cgn_groups,
        name="HOTimeChanCoupledGRUAugmented",
        **kwargs,
    ):
        super().__init__(name=name)
        self.h_size = h_size
        self.n_layers = n_layers
        self.lam_1 = lam_1
        self.outsize = outsize
        self.group_size = group_size
        self.group_hop = group_hop
        self.cgn_groups = cgn_groups

        self.in_layers = hk.Sequential(
            [
                hk.Conv1D(
                    output_channels=self.h_size,
                    kernel_shape=self.group_size,
                    stride=self.group_hop,
                    w_init=complex_xavier,
                    b_init=complex_zeros,
                    padding="VALID",
                ),
                complex_relu,
                CGN(groups=self.cgn_groups),
            ]
        )

        self.out_layers = hk.Sequential(
            [
                hk.Conv1DTranspose(
                    output_channels=self.outsize,
                    kernel_shape=self.group_size,
                    stride=self.group_hop,
                    w_init=complex_xavier,
                    b_init=complex_zeros,
                ),
            ]
        )

        self.rnn_stack = hk.DeepRNN(
            [
                CGRU(
                    hidden_size=self.h_size,
                    w_i_init=complex_xavier,
                    w_h_init=complex_xavier,
                    use_norm=False,
                )
                for _ in range(n_layers)
            ]
        )

    def preprocess_flatten(self, full_size_inputs, per_freq_inputs):
        # stack the gradients as well as any extra inputs
        input_stack = jnp.stack(full_size_inputs, axis=-1)

        # input will be T x F x M x .. reshape to F x T x M x ..
        input_stack_flat = jnp.swapaxes(input_stack, 0, 1)

        # flatten F x T x M x ... to F x TM..
        input_stack_flat = input_stack_flat.reshape((input_stack_flat.shape[0], -1))

        # stack in any final features
        input_stack_flat = jnp.concatenate(
            (input_stack_flat, *per_freq_inputs), axis=-1
        )

        # feature extraction options
        mag = jnp.log1p(jnp.abs(input_stack_flat))
        phase = jnp.exp(1.0j * jnp.angle(input_stack_flat))
        input_stack_flat = mag * phase

        # process and convert to F x H
        # padd for convolution
        input_stack_padded = jnp.pad(
            input_stack_flat,
            ((self.group_size // 2 + 1, self.group_size // 2 + 1), (0, 0)),
        )

        return self.in_layers(input_stack_padded)

    def postprocess_reshape(self, rnn_out, full_size_inputs):
        # process from F x H to F x TM
        out = -self.lam_1 * self.out_layers(rnn_out)

        # remove any zero padding
        F = full_size_inputs[0].shape[1]
        front_trim = jnp.ceil((out.shape[0] - F) / 2).astype(int)

        out = jax.lax.dynamic_slice(out, (front_trim, 0), (F, out.shape[1]))

        # rehape from F x TM to F x T x M
        out = out.reshape(
            out.shape[0], full_size_inputs[0].shape[0], full_size_inputs[0].shape[2]
        )

        # return to original T x F x M
        out = jnp.swapaxes(out, 0, 1)

        return out

    def __call__(self, full_size_inputs, h, per_freq_inputs):
        # do feature extraction like log1p and input coupling
        rnn_in = self.preprocess_flatten(full_size_inputs, per_freq_inputs)

        # take the RNN step
        rnn_out, h = self.rnn_stack(rnn_in, h)

        # reassemble to right shape and do any output coupling
        out = self.postprocess_reshape(rnn_out, full_size_inputs)

        return out, h

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("Optimizer")
        parser.add_argument("--h_size", type=int, default=32)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--outsize", type=int, default=1)
        parser.add_argument("--group_size", type=int, default=1)
        parser.add_argument("--group_hop", type=int, default=1)
        parser.add_argument("--cgn_groups", type=int, default=2)
        parser.add_argument("--lam_1", type=float, default=1e-2)
        parser.add_argument("--stop_grad_p", type=float, default=1e-2)
        parser.add_argument("--reset_noise_p", type=float, default=0.0)
        parser.add_argument("--reset_zero_p", type=float, default=0.0)
        parser.add_argument("--direct_pred", action="store_true")
        parser.add_argument("--no_inner_grad", action="store_true")

        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = [
            "h_size",
            "n_layers",
            "outsize",
            "group_size",
            "group_hop",
            "cgn_groups",
            "lam_1",
            "stop_grad_p",
            "reset_noise_p",
            "reset_zero_p",
            "direct_pred",
            "no_inner_grad",
        ]

        filled_in_keys = {}
        for k in keys:
            if k in kwargs:
                filled_in_keys[k] = kwargs[k]
            else:
                filled_in_keys[k] = HO_FGRU_AUG.default_args()[k]
        return filled_in_keys

    @staticmethod
    def default_args():
        return {
            "h_size": 32,
            "n_layers": 2,
            "outsize": 1,
            "group_size": 1,
            "group_hop": 1,
            "cgn_groups": 2,
            "lam_1": 1e-2,
            "stop_grad_p": 1e-2,
            "reset_noise_p": 0.0,
            "reset_zero_p": 0.0,
            "direct_pred": False,
            "no_inner_grad": False,
        }


def _fwd(full_size_inputs, h, per_freq_inputs, **kwargs):
    optimizer = HO_FGRU_AUG(**kwargs)
    return optimizer(full_size_inputs, h, per_freq_inputs)


def make_deep_coupled_initial_state(params, **kwargs):
    b_size = (
        params.shape[1]
        + (kwargs["group_size"] // 2 + 1) * 2
        - (kwargs["group_size"] - kwargs["group_hop"])
    ) // kwargs["group_hop"]
    h_size = kwargs["h_size"]
    n_layers = kwargs["n_layers"]

    def single_layer_initial_state():
        state = jnp.zeros([h_size], dtype=jnp.dtype("complex64"))
        state = add_batch(state, b_size)
        return state

    return tuple(single_layer_initial_state() for _ in range(n_layers))


def init_optimizer_all_data(filter_p, batch_data, optimizer_dict, key):
    single_p = jax.tree_util.tree_leaves(filter_p)[0]

    # use that parameter to make an optimizer state
    h = make_deep_coupled_initial_state(single_p, **optimizer_dict["optimizer_kwargs"])

    # now init the optimizer
    full_size_inputs = [single_p] * 2
    per_freq_inputs = [single_p[0, :, 0, None]] * 3

    # now init the optimizer
    optimizer_p = optimizer_dict["optimizer"].init(
        key,
        full_size_inputs,
        h,
        per_freq_inputs,
        **optimizer_dict["optimizer_kwargs"],
    )

    return optimizer_p


@optimizers.optimizer
def make_mapped_optimizer_all_data(
    optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs
):
    direct_pred = optimizer_kwargs["direct_pred"]
    no_inner_grad = optimizer_kwargs["no_inner_grad"]

    def init(filter_p):
        state = make_deep_coupled_initial_state(filter_p, **optimizer_kwargs)
        return (filter_p, state)

    def update(i, features, jax_state):
        filter_p, state = jax_state

        if no_inner_grad:
            f = features.filter_features
        elif "grad" in features.cur_outputs:
            f = features.cur_outputs["grad"]
        else:
            f = jnp.conj(features.filter_features)

        u = features.cur_outputs["u"]
        d = features.cur_outputs["d"][0]
        e = features.cur_outputs["e"][0]
        y = features.cur_outputs["y"][0]

        full_size_inputs = [u, f]
        per_freq_inputs = [d, e, y]

        update, state = optimizer.apply(
            optimizer_p,
            None,
            full_size_inputs,
            state,
            per_freq_inputs,
            **optimizer_kwargs,
        )

        if direct_pred:
            filter_p = update
        else:
            filter_p = filter_p + update

        return (filter_p, state)

    def get_params(jax_state):
        # state was (parameters, state)
        return jax_state[0]

    return init, update, get_params


@optimizers.optimizer
def make_train_mapped_optimizer_all_data(
    optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs
):
    stop_grad_p = optimizer_kwargs["stop_grad_p"]
    reset_noise_p = optimizer_kwargs["reset_noise_p"]
    reset_zero_p = optimizer_kwargs["reset_zero_p"]
    direct_pred = optimizer_kwargs["direct_pred"]
    no_inner_grad = optimizer_kwargs["no_inner_grad"]

    def init(filter_p):
        state = make_deep_coupled_initial_state(filter_p, **optimizer_kwargs)
        return (filter_p, state)

    def update(i, features, jax_state):
        key, subkey = jax.random.split(features.key)
        stop_grad = jax.random.uniform(subkey) < stop_grad_p

        jax_state = jax.lax.cond(
            stop_grad, jax.lax.stop_gradient, lambda x: x, jax_state
        )

        filter_p, state = jax_state

        key, subkey = jax.random.split(key)
        reset_noise = jax.random.uniform(subkey) < reset_noise_p
        key, subkey = jax.random.split(key)
        reset_zero = jax.random.uniform(subkey) < reset_zero_p

        filter_p = jax.lax.cond(
            reset_noise,
            lambda x: jax.random.normal(key, shape=x.shape)
            + jax.random.normal(key, shape=x.shape) * 1j,
            lambda x: x,
            filter_p,
        )

        filter_p = jax.lax.cond(reset_zero, jnp.zeros_like, lambda x: x, filter_p)

        if no_inner_grad:
            f = features.filter_features
        elif "grad" in features.cur_outputs:
            f = features.cur_outputs["grad"]
        else:
            f = jnp.conj(features.filter_features)

        u = features.cur_outputs["u"]
        d = features.cur_outputs["d"][0]
        e = features.cur_outputs["e"][0]
        y = features.cur_outputs["y"][0]

        full_size_inputs = [u, f]
        per_freq_inputs = [d, e, y]

        update, state = optimizer.apply(
            optimizer_p,
            None,
            full_size_inputs,
            state,
            per_freq_inputs,
            **optimizer_kwargs,
        )

        if direct_pred:
            filter_p = update
        else:
            filter_p = filter_p + update

        return (filter_p, state)

    def get_params(jax_state):
        # state was (parameters, state)
        return jax_state[0]

    return init, update, get_params
