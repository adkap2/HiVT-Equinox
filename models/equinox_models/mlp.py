import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, List
from jaxtyping import Array, Float, PRNGKeyArray, Int, Bool, Union, Scalar
from models.equinox_models.embedding import SingleInputEmbedding, MultipleInputEmbedding

from einops import rearrange, reduce

# Import beartype
from beartype import beartype
from typing import List, Tuple, Optional

from utils import print_array_type

# Add jax type signature to inputs and outputs


@beartype
class ReLU(eqx.Module):
    def __call__(self, x: Float[Array, "hidden_dim=8"], key=None):
        output = jax.nn.relu(x)  # Float[Array, "batch 8"]
        return output


@beartype
class MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    relu: ReLU

    # @beartype
    def __init__(self, embed_dim: int, dropout_rate: float, keys: PRNGKeyArray):
        self.linear1 = eqx.nn.Linear(embed_dim, embed_dim * 4, key=keys[0])
        self.linear2 = eqx.nn.Linear(embed_dim * 4, embed_dim, key=keys[1])
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)
        self.relu = ReLU()

    # @beartype
    def __call__(
        self,
        nodes: Float[Array, "node_dim=2"],
        key,  # Shape: [batch_size, node_dim=2] -> Float[Array, "2 2"]
    ) -> Float[
        Array, "node_dim=2"
    ]:  # Shape: [batch_size, node_dim=2] -> Float[Array, "2 2"]

        key1, key2 = jax.random.split(key)
        nodes = self.linear1(nodes)  # Float[Array, "batch=2 num_heads=8"]

        nodes = self.relu(nodes)  # Float[Array, "batch 8"]

        nodes = self.dropout1(nodes, key=key1)  # Float[Array, "batch 8"]

        nodes = self.linear2(nodes)  # Float[Array, "batch 2"]
        nodes = self.dropout2(nodes, key=key2)  # Float[Array, "batch node_dim=2"]
        return nodes
