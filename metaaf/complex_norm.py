import jax.numpy as jnp
import haiku as hk


class CGN(hk.Module):
    def __init__(
        self, groups=6, create_scale=True, create_offset=True, eps=1e-5, name=None
    ):
        super().__init__(name=name)
        self.gn_real = hk.GroupNorm(
            groups=groups,
            create_scale=create_scale,
            create_offset=create_offset,
            eps=eps,
        )
        self.gn_imag = hk.GroupNorm(
            groups=groups,
            create_scale=create_scale,
            create_offset=create_offset,
            eps=eps,
        )

    def __call__(self, x):
        x_real = jnp.real(x)
        x_imag = jnp.imag(x)

        x_real_n = self.gn_real(x_real)
        x_imag_n = self.gn_imag(x_imag)

        return (x_real_n + 1j * x_imag_n) / jnp.sqrt(2)


class CLNorm(hk.Module):
    def __init__(
        self,
        axis,
        create_scale,
        create_offset,
        eps=1e-05,
        scale_init=None,
        offset_init=None,
        use_fast_variance=False,
        name=None,
        param_axis=None,
    ):
        super().__init__(name=name)
        self.n_real = hk.LayerNorm(
            axis=axis,
            create_scale=create_scale,
            create_offset=create_offset,
            eps=eps,
            scale_init=scale_init,
            offset_init=offset_init,
            use_fast_variance=use_fast_variance,
            name=name,
            param_axis=param_axis,
        )

        self.n_imag = hk.LayerNorm(
            axis=axis,
            create_scale=create_scale,
            create_offset=create_offset,
            eps=eps,
            scale_init=scale_init,
            offset_init=offset_init,
            use_fast_variance=use_fast_variance,
            name=name,
            param_axis=param_axis,
        )

    def __call__(self, x):
        x_real = jnp.real(x)
        x_imag = jnp.imag(x)

        x_real_n = self.n_real(x_real)
        x_imag_n = self.n_imag(x_imag)

        return (x_real_n + 1j * x_imag_n) / jnp.sqrt(2)
