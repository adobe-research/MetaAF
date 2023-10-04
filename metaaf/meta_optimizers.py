# modified from https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/optimizers.html#adam
# Original license below.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax.numpy as jnp
from jax.example_libraries import optimizers


@optimizers.optimizer
def complex_adam(step_size, b1=0.9, b2=0.999, eps=1e-8):
    """Construct optimizer triple for Adam.

    Args:
    step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to a positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
        for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
        for the second moment estimates (default 0.999).
    eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).

    Returns:
    An (init_fun, update_fun, get_params) triple.
    """
    step_size = optimizers.make_schedule(step_size)

    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state):
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g.conj() * g) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        x = x - step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params


@optimizers.optimizer
def complex_adamw(step_size, b1=0.9, b2=0.999, l=1e-4, eps=1e-8):
    """Construct optimizer triple for Adamw.

    Args:
    step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to a positive scalar.
    b1: optional, a positive scalar value for beta_1, the exponential decay rate
        for the first moment estimates (default 0.9).
    b2: optional, a positive scalar value for beta_2, the exponential decay rate
        for the second moment estimates (default 0.999).
    l: optional, a positive scalar value for l2 regularization weight decay
    eps: optional, a positive scalar value for epsilon, a small constant for
        numerical stability (default 1e-8).

    Returns:
    An (init_fun, update_fun, get_params) triple.
    """
    step_size = optimizers.make_schedule(step_size)

    def init(x0):
        m0 = jnp.zeros_like(x0)
        v0 = jnp.zeros_like(x0)
        return x0, m0, v0

    def update(i, g, state):
        x, m, v = state
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g.conj() * g) + b2 * v  # Second moment estimate.
        mhat = m / (1 - jnp.asarray(b1, m.dtype) ** (i + 1))  # Bias correction.
        vhat = v / (1 - jnp.asarray(b2, m.dtype) ** (i + 1))
        update = step_size(i) * mhat / (jnp.sqrt(vhat) + eps)
        # add weight decay
        update = update + l * x
        x = x - update
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params
