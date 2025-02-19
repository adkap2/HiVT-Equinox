import equinox as eqx
import jax
import jax.numpy as jnp
from typing import List, Optional
from beartype import beartype
from jax._src.prng import (
    PRNGKeyArray,
)  # or from jaxlib.xla_extension import PrngKeyType
from utils import print_array_type


@beartype
class ReLU(eqx.Module):
    def __call__(self, x, key=None):  # Float[Array, "batch 2"]
        return jax.nn.relu(x)


class SingleInputEmbedding(eqx.Module):
    layers: list

    # @beartype
    # TODO FAILS BEARTYPE
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        *,
        n_layers: int = 3,
        key: jax.random.PRNGKey = None
    ) -> None:

        if key is None:
            key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, n_layers)

        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Linear(in_channel, out_channel, key=keys[0]),
                eqx.nn.LayerNorm(out_channel),  # Replace this with vmap
                ReLU(),
                eqx.nn.Linear(out_channel, out_channel, key=keys[1]),
                eqx.nn.LayerNorm(out_channel),
                ReLU(),
                eqx.nn.Linear(out_channel, out_channel, key=keys[2]),
                eqx.nn.LayerNorm(out_channel),
            ]
        )

    # Functions takes in an in channel dim vector and outputs an out channel dim vector
    @beartype
    def __call__(
        self, x: jnp.ndarray  # Float[Array, "batch 2"]
    ) -> jnp.ndarray:  # Add type signature
        return self.layers(x)


class MultipleInputEmbedding(eqx.Module):
    input_networks: list
    aggr_embed: eqx.Module

    # @beartype
    # TODO FAILS BEARTYPE
    def __init__(
        self,
        in_channels: list[int],  # Shape: [node_dim, edge_dim] -> [2, 2]
        out_channel: int,  # Shape: [embed_dim]
        *,
        key: jax.random.PRNGKey
    ) -> None:

        agg_key, *keys = jax.random.split(key, len(in_channels) + 1)

        # Create input networks
        self.input_networks = []
        for in_channel, key in zip(in_channels, keys):
            k1, k2 = jax.random.split(key)
            network = eqx.nn.Sequential(
                [
                    eqx.nn.Linear(in_channel, out_channel, key=k1),
                    eqx.nn.LayerNorm(out_channel),
                    ReLU(),
                    eqx.nn.Linear(out_channel, out_channel, key=k2),
                ]
            )
            self.input_networks.append(network)

        # Aggregation network
        k1, k2 = jax.random.split(agg_key)
        self.aggr_embed = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(out_channel),
                ReLU(),
                eqx.nn.Linear(out_channel, out_channel, key=k2),
                eqx.nn.LayerNorm(out_channel),
            ]
        )

    @beartype
    def __call__(
        self,
        continuous_inputs: List[jnp.ndarray],  # Shape: [batch_size, in_channel]
        categorical_inputs: Optional[
            List[jnp.ndarray]
        ] = None,  # Shape: [batch_size, in_channel]
    ) -> jnp.ndarray:  # Shape: [batch_size, out_channel]
        # print("EQUINOX continuous inputs :" , continuous_inputs)
        # TODO why is this different than torch?
        # Its a list but only contains two elements so specialize to that

        # if len(continuous_inputs) == 2:
        #     continuous_inputs = continuous_inputs[0], continuous_inputs[1]

        for i in range(len(self.input_networks)):
            continuous_inputs[i] = self.input_networks[i](continuous_inputs[i])

        output = jnp.sum(
            jnp.stack(continuous_inputs), axis=0
        )  # Float[Array, "batch 2"]

        if categorical_inputs is not None:
            output += jnp.stack(categorical_inputs).sum(axis=0)

        return self.aggr_embed(output)  # Float[Array, "batch 2"]
