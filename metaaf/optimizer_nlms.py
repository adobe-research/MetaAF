import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers


def add_args(parent_parser):
    return parent_parser


def grab_args(kwargs):
    return {}


def default_args():
    {}


def get_tuning_options(**kwargs):
    return {
        "step_size": jnp.logspace(-2, 1, 15),
        "u_forget_factor": [
            0.5,
            0.7,
            0.8,
            0.85,
            0.9,
            0.99,
            0.999,
        ],
    }


# this is a dummy function since we dont serialize anything
def _fwd(x, **kwargs):
    return x


def init_optimizer(filter_p, batch_data, optimizer_dict, key):
    # only needs to init the step size
    return {
        "step_size": 0.05,
        "u_forget_factor": 0.99,
    }


# the actual NLMS step
@optimizers.optimizer
def make_mapped_optmizer(optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs):
    step_size = optimizer_p["step_size"]
    u_forget_factor = optimizer_p["u_forget_factor"]

    def init(filter_p):
        # real valued since they store a magnitude
        u_norm = jnp.ones((1, filter_p.shape[1], filter_p.shape[2]))

        return (filter_p, u_norm)

    def update(i, features, jax_state):
        filter_p = jax_state[0]

        # get u normalization
        prev_u_norm = jax_state[1]
        u = features.cur_outputs["u"]
        u_norm = u_forget_factor * prev_u_norm + (1 - u_forget_factor) * (
            jnp.abs(u) ** 2
        ).sum(0, keepdims=True)

        # compute parameter updates
        update = (step_size * features.cur_outputs["grad"]) / u_norm

        # return updated params and updated normalizations
        return (filter_p + update, u_norm)

    def get_params(jax_state):
        return jax_state[0]

    return init, update, get_params
