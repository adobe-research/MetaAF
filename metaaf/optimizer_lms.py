import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers


def add_args(parent_parser):
    return parent_parser


def grab_args(kwargs):
    return {}


def default_args():
    return {}


def get_tuning_options(**kwargs):
    return {"step_size": jnp.logspace(2, -5, 75)}


# this is a dummy function since we dont serialize anything
def _fwd(x, **kwargs):
    return x


def init_optimizer(filter_p, batch_data, optimizer_dict, key):
    # only needs to init the step size
    return {"step_size": 0.1}


# the actual LMS step
@optimizers.optimizer
def make_mapped_optmizer(optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs):
    step_size = optimizer_p["step_size"]

    def init(filter_p):
        return filter_p

    def update(i, features, jax_state):
        filter_p = jax_state
        update = step_size * features.cur_outputs["grad"]
        return filter_p + update

    def get_params(jax_state):
        return jax_state

    return init, update, get_params
