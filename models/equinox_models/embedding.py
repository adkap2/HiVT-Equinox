import equinox as eqx
import jax
import jax.numpy as jnp
from typing import List, Optional
from beartype import beartype

@beartype
class ReLU(eqx.Module):
    def __call__(self, x, key=None):
        return jax.nn.relu(x)

# class LayerNorm(eqx.Module):
#     def __call__(self, x, key=None):
#         return eqx.nn.LayerNorm(x)

class SingleInputEmbedding(eqx.Module):
    layers: list
    
    # @beartype
    # TODO FAILS BEARTYPE
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 *,
                 n_layers: int = 3,
                 key: jax.random.PRNGKey = None) -> None:
        
        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, n_layers)
        
        self.layers = []
        for i, key in enumerate(keys):
            self.layers.append(eqx.nn.Sequential([
                eqx.nn.Linear(in_channel if i == 0 else out_channel, out_channel, key=key),
                # eqx.nn.LayerNorm(out_channel), # Replace this with vmap
                jax.vmap(eqx.nn.LayerNorm(out_channel)),
                # LayerNorm(),
                ReLU()
            ]))
    
    @beartype
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # print(f"x shape: {x.shape}")
        # print(f"x first few values: {x[:5]}")
        for layer in self.layers:
            x = layer(x)
        return x


# class MultipleInputEmbedding(eqx.Module):
#     input_networks: list
#     aggr_layers: list

    # # aggr_embed: eqx.Module
    
#     def __init__(self,
#                  in_channels: list[int],
#                  out_channel: int,
#                  *,
#                  key: jax.random.PRNGKey) -> None:
        
#         agg_key, *keys = jax.random.split(key, len(in_channels) + 1)
        
#         # Create input networks
#         self.input_networks = []
#         for in_channel, key in zip(in_channels, keys):
#             k1, k2 = jax.random.split(key)
#             network = [
#                 (eqx.nn.Linear(in_channel, out_channel, key=k1), eqx.nn.LayerNorm(out_channel)),
#                 (eqx.nn.Linear(out_channel, out_channel, key=k2), eqx.nn.LayerNorm(out_channel))
#             ]
#             self.input_networks.append(network)
        
#         # Aggregation network
#         k1, k2 = jax.random.split(agg_key)
#         self.aggr_layers = [
#             (eqx.nn.Linear(out_channel, out_channel, key=k1), eqx.nn.LayerNorm(out_channel)),
#             (eqx.nn.Linear(out_channel, out_channel, key=k2), eqx.nn.LayerNorm(out_channel))
#         ]
    
    # def __call__(self,
    #              continuous_inputs: List[jnp.ndarray], 
    #              categorical_inputs: Optional[List[jnp.ndarray]] = None) -> jnp.ndarray:
    #     # Process continuous inputs
    #     processed_inputs = []
    #     for x, network in zip(continuous_inputs, self.input_networks):
    #         # Apply each layer in the network
    #         for linear, norm in network:
    #             x = linear(x)
    #             # x = norm(x)
    #             x = jax.vmap(norm)(x)
    #             x = jax.nn.relu(x)
    #         processed_inputs.append(x)
        
    #     # Add categorical if present
    #     if categorical_inputs is not None:
    #         # jax.debug.print(categorical_inputs.shape)
    #         # jax.debug.print(x.shape)
    #         processed_inputs.extend(categorical_inputs)
        
    #     # Sum embeddings and apply aggregation
    #     output = jnp.sum(jnp.stack(processed_inputs), axis=0)
        
    #     print("EQUINOX output forward shape", output.shape)
    #     print("EQUINOX output forward first few values", output[:5])
        
    #     # Apply aggregation layers
    #     for linear, norm in self.aggr_layers:
    #         output = linear(output)
    #         # output = norm(output)  # Remove vmap here
    #         output = jax.vmap(norm)(output)
    #         output = jax.nn.relu(output)
            
    #     return output

from jax._src.prng import PRNGKeyArray  # or from jaxlib.xla_extension import PrngKeyType

class MultipleInputEmbedding(eqx.Module):
    input_networks: list
    aggr_embed: eqx.Module
    
    # @beartype
    # TODO FAILS BEARTYPE
    def __init__(self,
                 in_channels: list[int],
                 out_channel: int,
                 *,
                 key: jax.random.PRNGKey) -> None:
        
        agg_key, *keys = jax.random.split(key, len(in_channels) + 1)
        
        # Create input networks
        self.input_networks = []
        for in_channel, key in zip(in_channels, keys):
            k1, k2 = jax.random.split(key)
            network = eqx.nn.Sequential([
                eqx.nn.Linear(in_channel, out_channel, key=k1),
                lambda x, **kwargs: jax.vmap(eqx.nn.LayerNorm(out_channel))(x),  # Handle extra kwargs
                ReLU(),
                eqx.nn.Linear(out_channel, out_channel, key=k2),
                lambda x, **kwargs: jax.vmap(eqx.nn.LayerNorm(out_channel))(x)   # Handle extra kwargs
            ])
            self.input_networks.append(network)
        
        # Aggregation network
        k1, k2 = jax.random.split(agg_key)
        self.aggr_embed = eqx.nn.Sequential([
            eqx.nn.Linear(out_channel, out_channel, key=k1),
            lambda x, **kwargs: jax.vmap(eqx.nn.LayerNorm(out_channel))(x),  # Handle extra kwargs
            ReLU(),
            eqx.nn.Linear(out_channel, out_channel, key=k2),
            lambda x, **kwargs: jax.vmap(eqx.nn.LayerNorm(out_channel))(x)   # Handle extra kwargs
        ])


    @beartype
    def __call__(self,
                 continuous_inputs: List[jnp.ndarray], 
                 categorical_inputs: Optional[List[jnp.ndarray]] = None) -> jnp.ndarray:
        # Process continuous inputs
        processed_inputs = []
        for x, network in zip(continuous_inputs, self.input_networks):
            processed_inputs.append(network(x))
        
        output = jnp.stack(processed_inputs).sum(axis=0)
        # print("EQUINOX output forward shape", output.shape)
        # print("EQUINOX output forward first few values", output[:5])
        
        if categorical_inputs is not None:
            output += jnp.stack(categorical_inputs).sum(axis=0)
        return self.aggr_embed(output)