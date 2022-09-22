import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk

class CGN(hk.Module):
    def __init__(self, groups=6, create_scale=True, create_offset=True, eps=1e-5, name=None):
        super().__init__(name=name)
        self.gn_real = hk.GroupNorm(groups=groups, create_scale=create_scale, create_offset=create_offset, eps=eps)
        self.gn_imag = hk.GroupNorm(groups=groups, create_scale=create_scale, create_offset=create_offset, eps=eps)

    def __call__(self, x):
        x_real = jnp.real(x)
        x_imag = jnp.imag(x)
        
        x_real_n = self.gn_real(x_real)
        x_imag_n = self.gn_imag(x_imag)
        
        return (x_real_n + 1j * x_imag_n) / jnp.sqrt(2)