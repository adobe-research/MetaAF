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
    return {
        "step_size": jnp.logspace(0, -4, 25),
        "gamma": [0.7, 0.8, 0.9, 0.98, 0.99, 0.999],
    }


# this is a dummy function since we dont serialize anything
def _fwd(x, **kwargs):
    return x


def init_optimizer(filter_p, batch_data, optimizer_dict, key):
    # only needs to init the step size
    return {"step_size": 0.1, "gamma": 0.9, "eps": 1e-8}


# the actual rmsprop step
@optimizers.optimizer
def make_mapped_optmizer(optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs):
    def init(filter_p):
        return (filter_p, jnp.zeros_like(filter_p))

    def update(i, features, jax_state):
        filter_p, avg_sq_grad = jax_state
        g = jnp.conj(features.filter_features)

        gamma = optimizer_p["gamma"]
        eps = optimizer_p["eps"]

        avg_sq_grad = avg_sq_grad * gamma + jnp.square(jnp.abs(g)) * (1.0 - gamma)
        filter_p = filter_p - optimizer_p["step_size"] * g / jnp.sqrt(avg_sq_grad + eps)

        return (filter_p, avg_sq_grad)

    def get_params(jax_state):
        # state was (parameters, state)
        return jax_state[0]

    return init, update, get_params
