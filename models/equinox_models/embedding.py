import equinox as eqx
import jax
import jax.numpy as jnp
from typing import List, Optional

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


class MultipleInputEmbedding(eqx.Module):
    input_networks: list
    aggr_embed: tuple  # (LayerNorm, Linear, LayerNorm)
    
    def __init__(self,
                 in_channels: List[int],
                 out_channel: int,
                 *,
                 key: jax.random.PRNGKey) -> None:
        
        keys = jax.random.split(key, len(in_channels) + 1)
        
        # Create input networks matching PyTorch version
        self.input_networks = []
        for in_channel, key in zip(in_channels, keys[:-1]):
            k1, k2 = jax.random.split(key)
            network = (
                eqx.nn.Linear(in_channel, out_channel, key=k1),
                eqx.nn.LayerNorm(out_channel),
                eqx.nn.Linear(out_channel, out_channel, key=k2),
                eqx.nn.LayerNorm(out_channel)
            )
            self.input_networks.append(network)
        
        # Aggregation network
        k1, k2 = jax.random.split(keys[-1])
        self.aggr_embed = (
            eqx.nn.LayerNorm(out_channel),
            eqx.nn.Linear(out_channel, out_channel, key=k1),
            eqx.nn.LayerNorm(out_channel)
        )
    
    def __call__(self,
                 continuous_inputs: List[jnp.ndarray],
                 categorical_inputs: Optional[List[jnp.ndarray]] = None) -> jnp.ndarray:
        # Process continuous inputs
        processed_inputs = []
        for x, network in zip(continuous_inputs, self.input_networks):
            lin1, norm1, lin2, norm2 = network
            x = lin1(x)
            x = norm1(x)
            x = jax.nn.relu(x)
            x = lin2(x)
            x = norm2(x)
            processed_inputs.append(x)
            
        # Sum embeddings
        output = jnp.sum(jnp.stack(processed_inputs), axis=0)
        
        # Add categorical if present
        if categorical_inputs is not None:
            output += jnp.sum(jnp.stack(categorical_inputs), axis=0)
        
        # Final aggregation
        norm1, linear, norm2 = self.aggr_embed
        output = norm1(output)
        output = jax.nn.relu(output)
        output = linear(output)
        output = norm2(output)
        
        return output
    