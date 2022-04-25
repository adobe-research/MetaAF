import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers


def add_args(parent_parser):
    parser = parent_parser.add_argument_group("Optimizer")
    parser.add_argument("--error_aware", action="store_true")
    return parent_parser


def grab_args(kwargs):
    keys = [
        "error_aware",
    ]
    return {k: kwargs[k] for k in keys}


def default_args():
    {"error_aware": False}


def get_tuning_options(**kwargs):
    if kwargs["error_aware"]:
        return {
            "step_size": jnp.logspace(2, -2, 10),
            "u_forget_factor": [0.9, 0.99, 0.999],
            "e_forget_factor": [0.5, 0.75, 0.9],
        }
    else:
        return {
            "step_size": jnp.logspace(2, -3, 25),
            "u_forget_factor": [0.8, 0.9, 0.99, 0.999, 0.9999],
        }


# this is a dummy function since we dont serialize anything
def _fwd(x, **kwargs):
    return x


def init_optimizer(filter_p, batch_data, optimizer_dict, key):
    # only needs to init the step size
    return {"step_size": 0.05, "u_forget_factor": 0.99, "e_forget_factor": 0.7}


# the actual NLMS step
@optimizers.optimizer
def make_mapped_optmizer(optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs):
    step_size = optimizer_p["step_size"]
    u_forget_factor = optimizer_p["u_forget_factor"]

    error_aware = optimizer_kwargs["error_aware"]
    e_forget_factor = optimizer_p["e_forget_factor"]

    def init(filter_p):
        # real valued since they store a magnitude
        u_norm = jnp.ones(filter_p.shape)

        if error_aware:
            e_norm = jnp.zeros(filter_p.shape)
            return (filter_p, u_norm, e_norm)

        return (filter_p, u_norm)

    def update(i, features, jax_state):
        filter_p = jax_state[0]

        # get u normalization
        prev_u_norm = jax_state[1]
        u = features.cur_outputs["u"]
        u_norm = jnp.abs(u) ** 2
        u_norm = u_forget_factor * prev_u_norm + (1 - u_forget_factor) * u_norm

        # load into denominator
        total_norm = u_norm

        # get error normalization if needed and load into denominator
        if error_aware:
            prev_e_norm = jax_state[2]

            e = features.cur_outputs["e"]
            e_norm = (
                e_forget_factor * prev_e_norm + (1 - e_forget_factor) * jnp.abs(e) ** 2
            )

            total_norm = total_norm + 0.5 * e_norm

        # compute parameter updates
        update = -step_size * jnp.conj(features.filter_features) / total_norm

        # return updated params and updated normalizations
        if error_aware:
            return (filter_p + update, u_norm, e_norm)
        else:
            return (filter_p + update, u_norm)

    def get_params(jax_state):
        return jax_state[0]

    return init, update, get_params
