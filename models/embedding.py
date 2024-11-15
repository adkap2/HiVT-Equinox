import equinox as eqx
import jax
import jax.numpy as jnp

class SingleInputEmbedding(eqx.Module):
    layers: list
    
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 *,
                 key: jax.random.PRNGKey = None) -> None:
        
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)  # Need 3 keys for 3 linear layers
        
        # Create each layer component
        linear1 = eqx.nn.Linear(in_channel, out_channel, key=keys[0])
        #norm1 = eqx.nn.LayerNorm(out_channel)
        
        linear2 = eqx.nn.Linear(out_channel, out_channel, key=keys[1])
        #norm2 = eqx.nn.LayerNorm(out_channel)
        
        linear3 = eqx.nn.Linear(out_channel, out_channel, key=keys[2])
        #norm3 = eqx.nn.LayerNorm(out_channel)
        
        # Store as a list for sequential processing
        self.layers = [
            linear1,
            #norm1,
            jax.nn.relu,
            linear2,
            #norm2,
            jax.nn.relu,
            linear3,
            #norm3
        ]
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            if isinstance(layer, eqx.nn.Linear):
                # Use vmap for batch processing
                x = jax.vmap(layer)(x)
            elif isinstance(layer, eqx.nn.LayerNorm):
                # Apply layer norm to each item in batch
                x = jax.vmap(layer)(x)
            else:
                # ReLU can be applied directly
                x = layer(x)
        return x