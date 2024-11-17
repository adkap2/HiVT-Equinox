import equinox as eqx
import jax
import jax.numpy as jnp

class SingleInputEmbedding(eqx.Module):
    layers: list
    
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 *,
                 n_layers: int = 3,
                 key: jax.random.PRNGKey = None) -> None:
        
        if key is None:
            key = jax.random.PRNGKey(0)
        key, *keys = jax.random.split(key, n_layers)  # Need 3 keys for 3 linear layers
        
        self.layers = [(eqx.nn.Linear(in_channel, out_channel, key=key), eqx.nn.LayerNorm(out_channel))]
        for key in keys:
            self.layers.append((eqx.nn.Linear(out_channel, out_channel, key=key),
                                eqx.nn.LayerNorm(out_channel)))


    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for linear, norm in self.layers:
            x = linear(x)
            x = norm(x)
            x = jax.nn.relu(x)
        return x
