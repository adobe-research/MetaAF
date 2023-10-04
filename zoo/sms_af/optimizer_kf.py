import jax.numpy as jnp
from jax.example_libraries import optimizers


def add_args(parent_parser):
    parser = parent_parser.add_argument_group("Optimizer")
    return parent_parser


def grab_args(kwargs):
    return {}


def default_args():
    {}


def get_tuning_options(**kwargs):
    return {
        "init_scale_R": jnp.logspace(-4, 0, 3),
        "init_scale_P": jnp.logspace(-2, 0, 3),
        "forget_factor": [0.7, 0.9, 0.99],
        "regularization": jnp.concatenate((jnp.zeros(1), jnp.logspace(-8, -2, 3))),
    }


# this is a dummy function since we dont serialize anything
def _fwd(x, **kwargs):
    return x


def init_optimizer(filter_p, batch_data, optimizer_dict, key):
    # only needs to init the step size
    return {
        "forget_factor": 0.75,
        "init_scale_R": 1e-4,
        "init_scale_P": 1e-2,
        "regularization": 1e-2,
    }


# the actual KF step
@optimizers.optimizer
def make_mapped_optimizer(optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs):
    forget_factor = optimizer_p["forget_factor"]
    init_scale_R = optimizer_p["init_scale_R"]
    init_scale_P = optimizer_p["init_scale_P"]
    regularization = optimizer_p["regularization"]

    def init(filter_p):
        # real valued since they store a magnitude
        R = jnp.ones((1, filter_p.shape[1]), dtype=filter_p.dtype) * init_scale_R
        P = (
            jnp.ones((filter_p.shape[0], filter_p.shape[1]), dtype=filter_p.dtype)
            * init_scale_P
        )

        return (filter_p, R, P)

    def update(i, features, jax_state):
        filter_p, R, P = jax_state

        e = features.cur_outputs["e"][-1, None, :, 0]  # 1 x F
        u = features.cur_outputs["u"][..., 0]  # Block x F

        # update error psd estimate
        R = forget_factor * R + (1 - forget_factor) * jnp.abs(e) ** 2

        # get kalman gain
        denom = u * P * u.conj() + 2 * R + regularization

        K = P * u.conj() / denom

        # update P
        P = (1 - init_scale_P**2) * (1 - 0.5 * K * u) * P + init_scale_P**2 * (
            jnp.abs(filter_p[..., 0]) ** 2
        )

        # get step size
        denom = u * P * u.conj() + 2 * R + regularization
        step = (1 - init_scale_P) * P / denom

        # update filter
        filter_p = filter_p + step[..., None] * u[..., None].conj() * e[..., None]

        return (filter_p, R, P)

    def get_params(jax_state):
        return jax_state[0]

    return init, update, get_params
