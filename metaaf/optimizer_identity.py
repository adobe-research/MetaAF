import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers

# this is a dummy function since we dont serialize anything
def _fwd(x, **kwargs):
    return x


def init_optimizer(filter_p, batch_data, optimizer_dict, key):
    return {}


# the actual identity step
@optimizers.optimizer
def make_mapped_optmizer(optimizer={}, optimizer_p={}, optimizer_kwargs={}, **kwargs):
    def init(filter_p):
        return filter_p

    def update(i, features, jax_state):
        return jax_state

    def get_params(jax_state):
        return jax_state

    return init, update, get_params
