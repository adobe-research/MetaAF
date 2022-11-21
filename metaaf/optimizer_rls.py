import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

# arg parsing utilities
def add_args(parent_parser):
    parser = parent_parser.add_argument_group("Optimizer")
    parser.add_argument("--nrls", action="store_true")
    parser.add_argument("--optimize_conjugate", action="store_true")
    return parent_parser


def grab_args(kwargs):
    keys = [
        "nrls",
        "optimize_conjugate",
    ]
    return {k: kwargs[k] for k in keys}


def default_args():
    return {
        "nrls": False,
        "optimize_conjugate": True,
    }


def get_tuning_options(**kwargs):
    return {
        "init_scale": jnp.logspace(-2, 2, 10),
        "forget_factor": [
            0.5,
            0.7,
            0.75,
            0.8,
            0.9,
            0.99,
        ],
        "regularization": jnp.logspace(-5, 0, 3),
    }


# this is a dummy function since we dont serialize anything
def _fwd(x, **kwargs):
    return x


def init_optimizer(filter_p, batch_data, optimizer_dict, key):
    # only needs to init the step size
    return {
        "forget_factor": 0.999,
        "init_scale": 0.01,
        "regularization": 1e-4,
    }


def freq_flatten(x):
    # convert axis from Frame x Freq x Channels
    # to Freq x Frame * Channels
    x_swap = jnp.swapaxes(x, 0, 1)
    return x_swap.reshape((x_swap.shape[0], -1))


def freq_unflatten(x, orig_shape):
    # convert axis from Freq x Frame * Channels
    # to Frame x Freq x Channels
    x_unflatten = x.reshape((orig_shape[1], orig_shape[0], orig_shape[2]))
    return jnp.swapaxes(x_unflatten, 0, 1)


def update_kalman_gain(u, P, forget_factor, regularization):
    # Freq. x N
    P_u = jnp.einsum("fmn,fn->fm", P, u)

    # Freq.
    u_P_u = jnp.einsum("fm,fm->f", u.conj(), P_u)

    # dont divide by zero
    denom = forget_factor + u_P_u
    denom = jnp.maximum(denom, regularization * jnp.max(denom))
    return P_u / denom[:, None]


def update_covariance(u, P, K, forget_factor):
    # Freq. x N x N
    K_uh_P = jnp.einsum("fn,fm,fmi->fni", K, u.conj(), P)
    P = P - K_uh_P
    return P / forget_factor


def get_update(K, e):
    # assume only one channel output and index it
    update = jnp.einsum("fm,fn->fmn", K, e.conj())[..., 0]
    return update


def rls_step(P, u, e, forget_factor, regularization, normalize):
    # convert all inputs to flat form
    # e is already Freq. x Channels
    u_flat = freq_flatten(u)

    # if normalizing use power
    if normalize:
        power = jnp.mean(jnp.abs(u_flat) ** 2, 1) * forget_factor
    else:
        power = forget_factor

    # Compute the gain
    K = update_kalman_gain(u_flat, P, power, regularization)

    # Compute and update covariance
    P = update_covariance(u_flat, P, K, forget_factor)

    # update the filter
    w_flat_update = get_update(K, e)
    w_update = freq_unflatten(w_flat_update, u.shape)

    return w_update, P


# the actual RLS step
@optimizers.optimizer
def make_mapped_optmizer(optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs):
    forget_factor = optimizer_p["forget_factor"]
    init_scale = optimizer_p["init_scale"]
    regularization = (
        optimizer_p["regularization"] if "regularization" in optimizer_p else 1e-4
    )

    normalize = optimizer_kwargs["nrls"]
    optimize_conjugate = optimizer_kwargs["optimize_conjugate"]

    def init(filter_p):
        # filter_p is N x F x M we will track it as F x NM
        P = jnp.stack(
            [
                jnp.identity(
                    filter_p.shape[0] * filter_p.shape[2], dtype=filter_p.dtype
                )
                * init_scale
                for _ in range(filter_p.shape[1])
            ]
        )

        return (filter_p, P)

    def update(i, features, jax_state):
        w, P = jax_state
        e = features.cur_outputs["e"][0]
        u = features.cur_outputs["u"]

        w_update, P = rls_step(P, u, e, forget_factor, regularization, normalize)

        if optimize_conjugate:
            w = w + w_update.conj()
        else:
            w = w + w_update

        return (w, P)

    def get_params(jax_state):
        return jax_state[0]

    return init, update, get_params
