# modified to be complex-valued from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py#L515
# Original non-complex GRU code license below.

# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

from metaaf.complex_utils import (
    complex_variance_scaling,
    complex_zeros,
    complex_sigmoid,
    complex_tanh,
)
import types
from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union


class CGRU(hk.RNNCore):
    r"""Gated Recurrent Unit.
    The implementation is based on: https://arxiv.org/pdf/1412.3555v1.pdf with
    biases.
    Given :math:`x_t` and the previous state :math:`h_{t-1}` the core computes
    .. math::
     \begin{array}{ll}
     z_t &= \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z) \\
     r_t &= \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r) \\
     a_t &= \tanh(W_{ia} x_t + W_{ha} (r_t \bigodot h_{t-1}) + b_a) \\
     h_t &= (1 - z_t) \bigodot h_{t-1} + z_t \bigodot a_t
     \end{array}
    where :math:`z_t` and :math:`r_t` are reset and update gates.
    The output is equal to the new hidden state, :math:`h_t`.
    Warning: Backwards compatibility of GRU weights is currently unsupported.
    TODO(tycai): Make policy decision/benchmark performance for GRU variants.
    """

    def __init__(
        self,
        hidden_size: int,
        w_i_init: Optional[hk.initializers.Initializer] = None,
        w_h_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.w_i_init = w_i_init or complex_variance_scaling
        self.w_h_init = w_h_init or complex_variance_scaling
        self.b_init = b_init or complex_zeros
        self.sig = complex_sigmoid

    def __call__(self, inputs, state):
        if inputs.ndim not in (1, 2):
            raise ValueError("GRU input must be rank-1 or rank-2.")

        input_size = inputs.shape[-1]
        hidden_size = self.hidden_size
        w_i = hk.get_parameter(
            "w_i", [input_size, 3 * hidden_size], inputs.dtype, init=self.w_i_init
        )
        w_h = hk.get_parameter(
            "w_h", [hidden_size, 3 * hidden_size], inputs.dtype, init=self.w_h_init
        )
        b = hk.get_parameter("b", [3 * hidden_size], inputs.dtype, init=self.b_init)
        w_h_z, w_h_a = jnp.split(w_h, indices_or_sections=[2 * hidden_size], axis=1)
        b_z, b_a = jnp.split(b, indices_or_sections=[2 * hidden_size], axis=0)

        gates_x = jnp.matmul(inputs, w_i)

        zr_x, a_x = jnp.split(gates_x, indices_or_sections=[2 * hidden_size], axis=-1)
        zr_h = jnp.matmul(state, w_h_z)

        zr = zr_x + zr_h + jnp.broadcast_to(b_z, zr_h.shape)
        z, r = jnp.split(self.sig(zr), indices_or_sections=2, axis=-1)

        a_h = jnp.matmul(r * state, w_h_a)

        a = complex_tanh(a_x + a_h + jnp.broadcast_to(b_a, a_h.shape))

        next_state = (1 - z) * state + z * a

        return next_state, next_state

    def initial_state(self, batch_size: Optional[int]):
        state = jnp.zeros([self.hidden_size], dtype=jnp.complex64)
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state


def add_batch(nest, batch_size):
    # modified from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/recurrent.py
    """Adds a batch dimension at axis 0 to the leaves of a nested structure."""

    def broadcast(x):
        return jnp.broadcast_to(x, (batch_size,) + x.shape)

    return jax.tree_map(broadcast, nest)


def make_deep_initial_state(params, **kwargs):
    b_size = np.prod(params.shape)
    h_size = kwargs["h_size"]
    n_layers = kwargs["n_layers"]

    def single_layer_initial_state():
        state = jnp.zeros([h_size], dtype=np.dtype("complex64"))
        state = add_batch(state, b_size)
        return state

    return tuple(single_layer_initial_state() for _ in range(n_layers))
