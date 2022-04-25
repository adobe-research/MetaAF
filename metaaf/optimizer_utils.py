import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_map


def frame_indep_meta_mse(losses, outputs, data_samples, metadata, outer_learnable):
    return jnp.mean(losses)


def frame_indep_meta_logmse(losses, outputs, data_samples, metadata, outer_learnable):
    EPS = 1e-8
    return jnp.log(jnp.mean(losses) + EPS)


class FeatureContainer:
    def __init__(self, filter_features, cur_outputs, cur_inputs, metadata, key):
        # stop gradients w.r.t inputs for the optimizer
        self.cur_inputs = jax.lax.stop_gradient(cur_inputs)
        self.cur_outputs = jax.lax.stop_gradient(cur_outputs)
        self.filter_features = jax.lax.stop_gradient(filter_features)
        self.metadata = metadata
        self.key = key


# below is taken and modified from
# https://jax.readthedocs.io/en/latest/_modules/jax/example_libraries/optimizers.html#clip_grads
# Original license for all code below this line follows. 

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
def l2_norm(tree):
    """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
    leaves, _ = tree_flatten(tree)
    return jnp.sqrt(sum((jnp.abs(x) ** 2).sum() for x in leaves))


def clip_grads(grad_tree, max_norm, eps=1e-9):
    """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
    norm = l2_norm(grad_tree)

    def normalize(g):
        return jnp.where(norm < max_norm, g, g * max_norm / (norm + eps))

    return tree_map(normalize, grad_tree)
