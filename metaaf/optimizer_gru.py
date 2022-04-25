import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import haiku as hk
from metaaf.complex_gru import CGRU, make_deep_initial_state
from metaaf.complex_utils import complex_variance_scaling, complex_zeros, complex_relu


class ElementWiseGRU(hk.Module):
    def __init__(
        self,
        h_size,
        n_layers,
        lam_1,
        input_transform="log1p",
        name="ElementWiseGRU",
        **kwargs
    ):

        super().__init__(name=name)
        self.h_size = h_size
        self.n_layers = n_layers
        self.lam_1 = lam_1

        assert input_transform in ["raw", "log1p"]
        self.input_transform = input_transform

        self.in_lin = hk.Sequential(
            [
                hk.Linear(
                    output_size=self.h_size,
                    w_init=complex_variance_scaling,
                    b_init=complex_zeros,
                ),
                complex_relu,
            ]
        )

        self.out_lin = hk.Sequential(
            [
                hk.Linear(
                    output_size=self.h_size,
                    w_init=complex_variance_scaling,
                    b_init=complex_zeros,
                ),
                complex_relu,
                hk.Linear(
                    output_size=1,
                    w_init=complex_variance_scaling,
                    b_init=complex_zeros,
                ),
            ]
        )

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

        input_stack_flat = input_stack.reshape((-1, input_stack.shape[-1]))

        return self.in_lin(input_stack_flat)

    def postprocess_reshape(self, rnn_out, raw_input):
        reshaped_out = rnn_out.reshape((*raw_input.shape, rnn_out.shape[-1]))
        out = self.out_lin(reshaped_out)[..., 0]
        return -self.lam_1 * out

    def __call__(self, x, h, extra_inputs):
        # do feature extraction like log1p and input coupling
        rnn_in = self.preprocess_flatten(x, extra_inputs)

        # take the RNN step
        rnn_out, h = self.rnn_stack(rnn_in, h)

        # reassemble to right shape and do any output coupling
        out = self.postprocess_reshape(rnn_out, x)

        return out, h

    @staticmethod
    def add_args(parent_parser):
        parser = parent_parser.add_argument_group("Optimizer")
        parser.add_argument("--h_size", type=int, default=32)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--input_transform", type=str, default="log1p")
        parser.add_argument("--lam_1", type=float, default=1e-2)

        return parent_parser

    @staticmethod
    def grab_args(kwargs):
        keys = [
            "h_size",
            "n_layers",
            "lam_1",
            "input_transform",
        ]
        return {k: kwargs[k] for k in keys}

    @staticmethod
    def default_args():
        return {
            "h_size": 32,
            "n_layers": 2,
            "lam_1": 1e-2,
            "input_transform": "log1p",
        }


def _elementwise_gru_fwd(x, h, *extra_inputs, **kwargs):
    optimizer = ElementWiseGRU(**kwargs)
    return optimizer(x, h, extra_inputs)


def init_optimizer(filter_p, batch_data, optimizer_dict, key):
    single_p = jax.tree_util.tree_leaves(filter_p)[0]

    # use that parameter to make an optimizer state
    h = make_deep_initial_state(single_p, **optimizer_dict["optimizer_kwargs"])

    # now init the optimizer
    optimizer_p = optimizer_dict["optimizer"].init(
        key, single_p, h, **optimizer_dict["optimizer_kwargs"]
    )

    return optimizer_p


def init_optimizer_all_data(filter_p, batch_data, optimizer_dict, key):
    single_p = jax.tree_util.tree_leaves(filter_p)[0]

    # use that parameter to make an optimizer state
    h = make_deep_initial_state(single_p, **optimizer_dict["optimizer_kwargs"])

    # now init the optimizer
    optimizer_p = optimizer_dict["optimizer"].init(
        key,
        single_p,
        h,
        single_p,
        single_p,
        single_p,
        **optimizer_dict["optimizer_kwargs"],
    )

    return optimizer_p


@optimizers.optimizer
def make_mapped_optmizer(optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs):
    def init(filter_p):
        state = make_deep_initial_state(filter_p, **optimizer_kwargs)
        return (filter_p, state)

    def update(i, features, jax_state):
        filter_p, state = jax_state
        update, state = optimizer.apply(
            optimizer_p,
            None,
            jnp.conj(features.filter_features),
            state,
            **optimizer_kwargs,
        )

        return (filter_p + update, state)

    def get_params(jax_state):
        # state was (parameters, state)
        return jax_state[0]

    return init, update, get_params


@optimizers.optimizer
def make_mapped_optmizer_all_data(
    optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs
):
    def init(filter_p):
        state = make_deep_initial_state(filter_p, **optimizer_kwargs)
        return (filter_p, state)

    def update(i, features, jax_state):
        filter_p, state = jax_state

        u = features.cur_outputs["u"]
        d = jnp.broadcast_to(features.cur_outputs["d"], u.shape)
        e = jnp.broadcast_to(features.cur_outputs["e"], u.shape)

        update, state = optimizer.apply(
            optimizer_p,
            None,
            jnp.conj(features.filter_features),
            state,
            u,
            d,
            e,
            **optimizer_kwargs,
        )

        return (filter_p + update, state)

    def get_params(jax_state):
        # state was (parameters, state)
        return jax_state[0]

    return init, update, get_params
