import jax
import jax.numpy as jnp


def l2_normalize(x, axis=-1, epsilon=1e-12):
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))

    return x * x_inv_norm


def logdet(x):
    raise NotImplementedError


def saturate_cast(x, dtype):
    raise NotImplementedError
