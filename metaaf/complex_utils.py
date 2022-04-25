import jax
import jax.numpy as jnp
import haiku as hk


def complex_zeros(shape, _):
    return jnp.zeros(shape, dtype=jnp.complex64)


# see https://openreview.net/attachment?id=H1T2hmZAb&name=pdf
def complex_variance_scaling(shape, dtype):
    real = hk.initializers.VarianceScaling()(shape, dtype=jnp.float32)
    imag = hk.initializers.VarianceScaling()(shape, dtype=jnp.float32)

    mag = jnp.sqrt(real ** 2 + imag ** 2)
    angle = hk.initializers.RandomUniform(minval=-jnp.pi, maxval=jnp.pi)(
        shape, dtype=jnp.float32
    )

    return mag * jnp.exp(1j * angle)


def complex_sigmoid(x):
    return jax.nn.sigmoid(x.real + x.imag)


def complex_tanh(x):
    return jnp.tanh(x.real) + 1j * jnp.tanh(x.imag)


def complex_relu(x):
    return jax.nn.relu(x.real) + 1j * jax.nn.relu(x.imag)
